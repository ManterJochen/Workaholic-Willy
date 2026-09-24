"""Universal Robots :class:`RobotArm` implementation."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import re
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import numpy as np

from src.config.schema.robot import RobotConfig
from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, FrameMismatchError, Pose
from src.geometry.quaternion import angle_between
from src.robot.constants import HOME_JOINTS_DEFAULT, UR_ARM_LOG_FILE, create_robot_logger
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    CameraWorldDecline,
    CameraWorldStamp,
    ControllerPayload,
    DigitalIOPort,
    FreedriveSession,
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotKinematicsError,
    RobotMode,
    RobotMotionRejected,
    RobotStatus,
    SafetyMode,
    Wrench,
    active_decline,
    camera_world_refusal,
    resolve_camera_world,
    stamp_result,
)
from src.robot.core.arm_capabilities import LineMotion, LineReading, PayloadModel
from src.robot.core.camera_world import without_camera_world as _without_camera_world
from src.robot.safety import (
    LineSamples,
    PathSamples,
    SafetyPreflight,
    line_samples,
)
from src.robot.safety._ur_ik import NearestGoal, URBranch, nearest_goals, nearest_turn, ur_branch, ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm, ur_series_twin
from src.robot.safety.path_samples import (
    LINE_MAX_CHORD_OFF_MM,
    LINE_MAX_JOINT_STEP_RAD,
    MAX_PATH_SAMPLES,
    joint_path_samples,
    span_overshoot_rad,
)
from src.robot.safety.planning import (
    CuroboPlanClient,
    CuroboUnavailableError,
    StateRefusalKind,
    StateWhere,
)
from src.robot.safety.planning.hand import planner_hand
from src.robot.safety.workspace import WorkspaceGuard

from .connection import URConnection
from .freedrive import HAND_GUIDED_MESSAGE, URFreedriveSession, raise_unless_operational
from .motion import MotionController
from .pose import URPose
from .pose_adapter import pose_to_urpose, urpose_to_pose

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Callable, Sequence
    from typing import Any

    from src.robot.safety.planning.curobo_client import SidecarIdentity
    from src.robot.safety.planning.live_world import (
        LivePlannerWorld,
        WorldRefresh,
    )
    from src.robot.safety.planning.perceived import SelfEnvelope

    from .curobo_motion import CuroboUrPlanner

__all__ = ["UR_CAPABILITIES", "URRobotArm", "ur_capabilities"]


def ur_capabilities(model: str = "ur5e") -> RobotCapabilities:
    """The capability surface of this vendor for ``model``.

    The capability record is what downstream code reads to learn which robot it is
    holding, so a hardcoded "ur5e" would make every model-keyed lookup, the DH table,
    the mesh bundle and the cuRobo config, resolve the wrong robot on a real UR3e.
    Everything except the model string is shared across the e-series.
    """
    return dataclasses.replace(UR_CAPABILITIES, model=model)


UR_CAPABILITIES = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=True,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=True,  # backed: get_tcp_wrench / get_joint_torques via SupportsForceTorque
    is_simulated=False,
)

# UR controller mode integers to the vendor-neutral enums. UR getRobotMode runs from -1
# for no controller and 0 for disconnected up to 8 for updating firmware; getSafetyMode
# runs from 1 for normal to 9 for fault, and 10 to 13, the validate, undefined, auto-mode
# and three-position stops, map to unknown here. An unmapped integer falls back to
# unknown.
_UR_ROBOT_MODE: dict[int, RobotMode] = {
    0: RobotMode.DISCONNECTED, 1: RobotMode.CONFIRM_SAFETY, 2: RobotMode.BOOTING,
    3: RobotMode.POWER_OFF, 4: RobotMode.POWER_ON, 5: RobotMode.IDLE,
    6: RobotMode.BACKDRIVE, 7: RobotMode.RUNNING, 8: RobotMode.UPDATING_FIRMWARE,
}
_UR_SAFETY_MODE: dict[int, SafetyMode] = {
    1: SafetyMode.NORMAL, 2: SafetyMode.REDUCED, 3: SafetyMode.PROTECTIVE_STOP,
    4: SafetyMode.RECOVERY, 5: SafetyMode.SAFEGUARD_STOP, 6: SafetyMode.SYSTEM_EMERGENCY_STOP,
    7: SafetyMode.ROBOT_EMERGENCY_STOP, 8: SafetyMode.VIOLATION, 9: SafetyMode.FAULT,
}

#: Why a cuRobo ``move`` says MISSING while no live camera world is wired to the arm.
_UR_NO_LIVE_WORLD = "robot.ur.motion_planner is 'curobo' and no live camera world is wired to this arm"


def _planner_refusal_status(refusal: object | None) -> MotionStatus:
    """The status a cuRobo refusal reads as, by what the sidecar found.

    A joint outside the planner's limits is JOINT_LIMIT_REJECTED. The robot touching itself and the
    robot reaching into the planner's world are both SELF_COLLISION_REJECTED, the status whose
    meaning covers the declared fixtures, because :class:`MotionStatus` has no status for the world.
    A refusal the sidecar did not type keeps SELF_COLLISION_REJECTED, what every refusal read as
    before the kind was read.
    """
    if refusal is not None and getattr(refusal, "kind", None) == StateRefusalKind.JOINT_LIMIT:
        return MotionStatus.JOINT_LIMIT_REJECTED
    return MotionStatus.SELF_COLLISION_REJECTED


def _with_refusal(sentence: str, refusal: object | None) -> str:
    """``sentence``, followed by the sidecar's own account of the refusal where it gave one."""
    render = getattr(refusal, "render", None)
    return f"{sentence}. {render()}" if callable(render) else sentence


def _branch_text(branch: "URBranch | None") -> str:
    """A branch as a log line reads it, or that it has none, for a model with no DH chain."""
    return f"branch {branch.render()}" if branch is not None else "branch unknown"


def _same_branch(held: "URBranch | None", other: "URBranch | None") -> bool:
    """Whether ``other`` holds the arm as ``held`` does; an unknown branch, on a model with no DH chain, agrees."""
    return held is None or other is None or held.agrees_with(other)


@dataclasses.dataclass(frozen=True, slots=True)
class _Route:
    """The judged joint waypoints one move runs, and how they were chosen.

    ``waypoints`` starts where the arm stands (a plan's own start, which is where the planner read the arm) and ends
    on the goal; every leg between two neighbours was judged by both authorities. It is the very object they judged,
    never a copy, and it is run as it is. ``dense`` is how many waypoints cuRobo returned, two for a straight line.
    ``how`` is the route in words for the log, ``planned`` whether cuRobo planned it, which decides how a joint move
    runs it: a straight line is one ``moveJ`` to the goal, a plan one ``moveJ`` per waypoint.
    """

    waypoints: "Sequence[Sequence[float]]"
    dense: int
    how: str
    planned: bool


class URRobotArm(RobotArm):
    """The UR-backed arm driver.

    It satisfies the vendor-neutral :class:`RobotArm` Protocol and keeps the UR
    convenience API the current pipelines use.
    """

    def __init__(
        self,
        config: RobotConfig,
        home_joints: list[float] | None = None,
        *,
        curobo_client_factory: Callable[[], CuroboPlanClient] | None = None,
    ) -> None:
        self.config = config
        self.logger = create_robot_logger("URRobotArm", UR_ARM_LOG_FILE)

        self._conn = URConnection(
            ip=config.ur.ip,
            vel=config.motion_limits.max_velocity,
            acc=config.motion_limits.max_acceleration,
            frequency=config.ur.rtde_frequency,
        )
        self._guard = WorkspaceGuard(config.workspace_limits)
        self._motion = MotionController(
            self._conn,
            self._guard,
            max_velocity=config.motion_limits.max_velocity,
            max_acceleration=config.motion_limits.max_acceleration,
        )
        # The vendor-neutral safety preflight pipeline. It owns its own WorkspaceGuard,
        # shrunk by the configured ``limits.workspace_margin_mm``, so the typed
        # ``move()`` path enforces the tighter operator-defined margin while
        # ``MotionController`` keeps the full-box guard as a backstop for the bool path.
        self._preflight = SafetyPreflight.from_safety_config(
            config.safety, config.workspace_limits,
            # The hand the cell names and the arm the guard places it on. An exact mesh guard with no
            # hand named refuses here, at build, rather than checking whatever hand the arm bundle carries.
            hand=planner_hand(config), arm_model=config.ur.model,
        )
        # An explicit argument wins over the cell own config, which wins over the
        # UR5e-authored constant. A real cell that declares nothing still gets a home
        # pose, and move_home warns about it.
        self._home_joints = (
            list(home_joints) if home_joints
            else list(config.home_joint_positions or HOME_JOINTS_DEFAULT)
        )
        self._capabilities = ur_capabilities(config.ur.model)
        # The real-UR cuRobo binding. "ik", the default, uses the controller IK path.
        # "curobo" plans a collision-free trajectory through safety.planning and executes
        # it over ur_rtde, fail-closed.
        self._motion_planner = config.ur.motion_planner
        #: The part the gripper carries while one is attached, as (length_mm, lateral_margin_mm) from
        #: planning_world.payload. The self filter grows the hand by it; None when nothing is attached.
        self._attached_payload: tuple[float, float] | None = None
        #: Whether the planner took the part the last attach handed it.
        self._payload_in_planner = False
        self._curobo_client_factory = curobo_client_factory
        self._curobo_ur: CuroboUrPlanner | None = None
        #: The flange to TCP the controller applied when this arm last connected, derived by the tool
        #: frame check on a ``polyscope`` cell; None before a connect and after a disconnect.
        self._observed_tool_frame: "np.ndarray | None" = None
        #: The cell as the cameras see it, asked immediately before every plan. Handed in by
        #: whatever composes the cell rather than built here: a driver may not reach up into
        #: perception, and this one only ever asks. `None` is the unchanged path, where the
        #: declared world is registered once and the planner keeps it for the life of the cell.
        self._live_world: "LivePlannerWorld | None" = None
        #: The hand-guiding session whose with-block is running, or None. Every motion verb refuses while
        #: one is (:meth:`_hand_guided_refusal`).
        self._freedrive: URFreedriveSession | None = None

    # ------------------------------------------------------------------
    # Tool frame (flange <-> TCP)
    # ------------------------------------------------------------------
    #
    # Every Pose crossing the RobotArm Protocol is the TCP, meaning the grasp centre.
    # What the controller calls the TCP depends on its tool register, and
    # ``gripper.tool_frame.source`` declares who owns that fact:
    #   "willy"     the controller runs a bare flange and this driver composes the
    #               transform, as the Isaac driver does. No vendor SDK call is involved.
    #   "polyscope" the controller holds it, and poses pass through untouched.
    # Either way connect() derives what the controller is really using and refuses a
    # mismatch.

    @property
    def safety_preflight(self) -> "SafetyPreflight | None":
        """The guard pipeline that judges every motion commanded through this arm's own verbs.

        What it judges differs by command, and the difference is what a caller needs to
        know. A Cartesian command is judged at its target pose; on a cuRobo path the
        planned final configuration is judged without IK quality and motion continuity.
        A joint command is judged at its target configuration by the destination guards
        only (joint limits, self-collision, payload). The middle of a planned path is
        judged sample by sample, always, whenever a planner produced one, which is what
        :attr:`plans_paths` answers. The raw transports :attr:`connection` and
        :attr:`motion` reach the controller below it.

        It implements :class:`~src.robot.safety.attestation.SafetyGated`, so a caller
        asks what this arm will refuse without reaching into `_preflight`. That matters
        for the library API: a caller supplies their own arm, which enumerating the
        driver registry cannot see, so the answer travels with the object.
        """
        return self._preflight

    @property
    def home_joint_positions(self) -> tuple[float, ...]:
        """The joints :meth:`move_home` drives to, in radians: the constructor's, else the config's, else the default.

        The joint-limit guard reads it where ``safety.joint_limits.within_half_turn_of_home`` is set,
        and centres its window on it, so the window is about the home this arm actually goes to.
        """
        return tuple(float(v) for v in self._home_joints)

    @property
    def plans_paths(self) -> bool:
        """Whether a move on this arm produces a path, and so has a middle to judge.

        Read off the arm as it is now, never off a key somebody could set: a cuRobo move
        hands over a trajectory and every configuration of it is judged, while an ik move
        interpolates and has no middle anybody could look at. The attestation prints one of
        the two, so a cell cannot claim a path check it will never run, or hide one it
        always does.
        """
        return self._motion_planner == "curobo"

    def _declared_tool_matrix(self) -> "np.ndarray | None":
        """The declared flange-to-TCP 4x4, whoever applies it. ``None`` while it is undeclared.

        Two consumers want different things and the difference is not cosmetic:
        :meth:`_pose_to_controller` needs it in ``willy`` mode alone, because in
        ``polyscope`` the controller applies it, while the cuRobo path needs it always,
        which :meth:`_pose_to_flange` sets out.
        """
        tf = self.config.gripper.tool_frame
        if tf.source == "undeclared":
            return None
        from .tool_frame import tool_frame_matrix

        return tool_frame_matrix(tf.offset_mm, tf.rotation_quat_xyzw)

    def _pose_to_flange(self, pose: Pose) -> Pose:
        """TCP to flange, meaning tool0, using the declared transform whatever ``source`` says.

        cuRobo is why this is unconditional. Its kinematic model ends at ``tool0`` by
        construction: ``scripts/curobo/build_ur_config.py`` attaches the 2F-85 collision
        spheres to ``tool0`` and the planner plans from tool0 to the pose. So it never
        consults the controller tool register, and ``polyscope`` mode does not reach it.
        Handing cuRobo the grasp centre as a tool0 goal is wrong twice over: it drives
        the flange to the grasp point, which is the 132 mm error, and it hangs the
        gripper collision spheres a whole tool length past the target, so the
        collision-free plan is computed for a gripper that is not where the planner
        thinks it is.
        """
        t = self._declared_tool_matrix()
        if t is None:
            return pose
        from src.geometry.matrix import invert_homogeneous

        m = np.asarray(pose.to_matrix(), dtype=np.float64) @ invert_homogeneous(t)
        return Pose.from_matrix(m, frame=pose.frame, label=pose.label)

    def _pose_from_controller(self, urpose: URPose, *, label: str) -> Pose:
        """Controller frame -> TCP. Identity unless this driver owns the tool frame."""
        pose = urpose_to_pose(urpose, frame=Frame.BASE, label=label)
        t = self._declared_tool_matrix() if self.config.gripper.tool_frame.source == "willy" else None
        if t is None:
            return pose
        m = np.asarray(pose.to_matrix(), dtype=np.float64) @ t
        return Pose.from_matrix(m, frame=Frame.BASE, label=label)

    def _pose_to_controller(self, pose: Pose) -> URPose:
        """TCP -> the frame the controller expects. Identity unless this driver owns the tool frame."""
        if self.config.gripper.tool_frame.source != "willy":
            return pose_to_urpose(pose)   # polyscope applies it controller-side; undeclared never moves
        return pose_to_urpose(self._pose_to_flange(pose))

    #: The size classes this driver compares across the two spellings a controller and a
    #: config use. It is the digits alone on purpose: the URSim e-Series container
    #: reports ``UR3`` for a UR3e, so a string comparison against ``ur3e`` would refuse a
    #: correctly configured cell.
    _MODEL_SIZES = ("3", "5", "10", "16", "20", "30")

    @classmethod
    def _model_size(cls, name: str | None) -> str | None:
        """Turn ``'UR3'``, ``'ur3e'`` or ``'UR5e CB3'`` into the size digits, or ``None``.

        It takes the number that follows the ``UR`` and not every digit in the string.
        Concatenating all of them looks equivalent and is not: ``'UR5e CB3'`` yields
        ``'53'``, which matches no size and disables the check silently. Any string
        without a readable ``UR<n>`` returns ``None``, which the caller treats as no
        evidence.
        """
        if not name:
            return None
        match = re.search(r"ur\s*(\d+)", str(name), flags=re.IGNORECASE)
        if match is None:
            return None
        size = match.group(1)
        return size if size in cls._MODEL_SIZES else None

    def _verify_controller_model(self) -> None:
        """Refuse where the controller says it is a different size of robot from ``ur.model``.

        ``ur.model`` keys the safety DH chain, the exact-mesh collision bundle and the
        cuRobo robot config. A UR3e planned against UR5e link lengths, a2 and a3 at
        -243.55 and -213.2 mm against -425 and -392.2, is precisely the failure that
        field exists to prevent, and without this comparison the mismatch surfaces only
        downstream, as a tool-frame refusal quoting a distance nobody can act on.

        It fails closed on evidence and open on its absence. A dashboard that will not
        answer, or answers something this driver cannot parse, yields ``None`` and is
        logged rather than refused, because a false refusal at connect is not a safe
        failure but a cell that will not start, and the tool-frame check downstream
        still catches the dangerous case on its own. Only a parsed, positive
        disagreement refuses.

        The comparison is on the size class alone. The URSim e-Series container reports
        ``UR3`` for a UR3e, so comparing whole strings would refuse a correct cell.
        """
        declared = self._model_size(self._capabilities.model)
        reported_raw = self._conn.controller_model()
        reported = self._model_size(reported_raw)
        if declared is None or reported is None:
            self.logger.info(
                "controller model not compared: config says %r, controller says %r; one of the two "
                "is not a size this driver recognises, so there is no evidence of a mismatch",
                self._capabilities.model, reported_raw,
            )
            return
        if declared == reported:
            # Say what was not checked. A size class is not a model: a UR3 and a UR3e are both
            # "UR3" to this comparison, on purpose, because the controller reports the same
            # string for both. Logging a bare "verified" for a pair this check cannot separate
            # reads as an all-clear it did not earn, and the separation is done further down by
            # _verify_tool_frame, relatively rather than by a threshold.
            twin = ur_series_twin(self._capabilities.model)
            if twin is None:
                self.logger.info(
                    "controller model verified: config %r matches the controller's %r",
                    self._capabilities.model, reported_raw,
                )
            else:
                self.logger.info(
                    "controller SIZE verified: config %r and the controller's %r are both a "
                    "UR%s. The series was NOT checked here and cannot be from this string, "
                    "because a %s controller reports %r too. The tool-frame check next tells "
                    "the two apart by asking which of them explains the controller better.",
                    self._capabilities.model, reported_raw, declared, twin, reported_raw,
                )
            return
        raise RobotConnectionError(
            f"ur.model is {self._capabilities.model!r}, but this controller reports "
            f"{reported_raw!r}: a UR{reported} is not a UR{declared}. That field keys the safety DH "
            f"chain, the collision mesh bundle and the cuRobo robot config, so every pose this stack "
            f"plans would be computed against another robot's link lengths. Point ur.model at the arm "
            f"that is actually plugged in."
        )

    def _verify_tool_frame(self) -> None:
        """Refuse to stay connected where the controller actual tool frame is not the assumed one.

        It is derived and never read:
        ``inv(base to flange from the bundled DH table) @ getForwardKinematics(q)``.
        Both halves are already on the safety-critical path, so this needs no SDK symbol
        ``move()`` does not already require. The two modes predict different answers,
        near identity for "willy", where the controller must be bare, and the declared
        transform for "polyscope", so a pass also proves which mode the cell is in
        rather than merely that some number matched.
        """
        from .tool_frame import compare_tool_frames, derive_active_tool_frame, tool_frame_matrix

        tf = self.config.gripper.tool_frame
        declared = tool_frame_matrix(tf.offset_mm, tf.rotation_quat_xyzw)
        # "willy" composes on this side, so the controller itself carries no tool.
        expected = np.eye(4, dtype=np.float64) if tf.source == "willy" else declared

        # The no-argument FK, and why. `getForwardKinematics(q)` is corrupted by any
        # preceding moveJ or moveL, because it shares the controller float registers with
        # them. Reproduced end to end: the first connect verifies at 0.00 mm, the cell
        # moves once, and the next connect is refused with a delta over a metre, turning
        # a perfectly good arm away on the second run through the check that exists to
        # catch a wrong TCP. The no-argument form is immune, at 0.000 mm in every state
        # tried, and it is also exactly what this check wants: the frame the controller
        # is applying right now.
        #
        # Steadiness is then required rather than assumed, because the pose and the
        # joints are read at two instants. A moving arm would pair a pose with joints it
        # no longer has and refuse a good cell, and a false refusal at connect is not a
        # safe failure but a cell that will not start.
        if not self._conn.is_steady():
            raise RobotConnectionError(
                "cannot verify the tool frame while the arm is moving: the check pairs the joint "
                "vector with the controller's current pose, and those must describe the same "
                "instant. Let the arm come to rest and connect again."
            )
        joints = list(self._conn.get_joint_positions())
        controller_tcp = URPose.from_ur_list(self._conn.fk_current()).to_T()
        observed = derive_active_tool_frame(self._capabilities.model, joints, controller_tcp)
        if observed is None:
            raise RobotConnectionError(
                f"cannot verify the tool frame: no bundled DH table for ur.model "
                f"{self._capabilities.model!r}, so the controller's active tool frame is not "
                f"derivable. Set a supported model, or gripper.tool_frame.source: undeclared while "
                f"this cell is being built (which refuses to connect for a different, louder reason)."
            )
        d_t, d_r = compare_tool_frames(observed, expected)
        self._refuse_if_the_series_twin_fits_better(joints, controller_tcp, expected, d_t)
        self._observed_tool_frame = None
        if d_t > tf.verify_tolerance_mm or d_r > 5.0:
            # No disconnect here, because connect() rolls the connection back for any
            # failure of this check.
            other = "polyscope" if tf.source == "willy" else "willy"
            raise RobotConnectionError(
                f"gripper.tool_frame.source is {tf.source!r}, which expects the controller's tool "
                f"register to hold {'nothing (a bare flange)' if tf.source == 'willy' else 'the declared transform'}"
                f", but the controller is actually running a tool frame {d_t:.1f} mm / {d_r:.1f} deg "
                f"away from that (tolerance {tf.verify_tolerance_mm:.1f} mm). Commanding motion now "
                f"would drive the arm off by that much on every pose. Either the pendant's TCP is not "
                f"what this config says, or the cell is really in {other!r} mode, or ur.model "
                f"({self._capabilities.model!r}) is not the arm on the other end. This is derived "
                f"from THAT model's DH table, so a wrong model produces a wrong distance here. "
                f"Derived as inv(DH flange) @ getForwardKinematics; no PolyScope read required."
            )
        # A wrist camera placed from a recorded frame is stale once the controller applies another
        # one. Checked before the frame is kept, so a refused connect keeps nothing.
        if tf.source == "polyscope":
            for body in self._preflight.wrist_bodies(self):
                stale = body.record_refusal("polyscope", observed)  # type: ignore[attr-defined]
                if stale is not None:
                    raise RobotConnectionError(stale)
        self._observed_tool_frame = np.asarray(observed, dtype=np.float64).copy()
        self.logger.info(
            "tool frame verified: source=%s, controller matches within %.2f mm / %.2f deg",
            tf.source, d_t, d_r,
        )

    @property
    def active_tool_frame(self) -> "np.ndarray | None":
        """The flange to TCP :meth:`get_tcp_pose` applies now, as a 4x4 in millimetres, or None where
        it is not known.

        On a ``willy`` cell the driver composes the declared frame itself, so it is the declared matrix,
        connected or not. On a ``polyscope`` cell the controller applies its own tool setting, so it is
        the frame the tool frame check derived from the controller at the last connect, and None before
        a connect and after a disconnect. An ``undeclared`` cell has none. An eye in hand calibration
        records this beside its solve.
        """
        source = self.config.gripper.tool_frame.source
        if source == "willy":
            return self._declared_tool_matrix()
        if source == "polyscope" and self._observed_tool_frame is not None:
            return self._observed_tool_frame.copy()
        return None

    #: How much better the twin has to explain the controller before this refuses, in mm. Small
    #: on purpose: the two hypotheses are not close. A correct cell reads about 0 mm for its own
    #: model and 8.5 mm or more for the twin (the measured ur3/ur3e minimum over 20000 poses),
    #: so anything above sensor noise separates them. This is NOT a tolerance on being right; it
    #: is the margin by which the wrong answer has to win before the cell is refused.
    _TWIN_MARGIN_MM = 1.0

    def _refuse_if_the_series_twin_fits_better(
        self, joints: list[float], controller_tcp, expected, own_d_t: float
    ) -> None:
        """Refuse where the other series of this size explains the controller better than this one does.

        The check the tolerance cannot do. ``_verify_controller_model`` compares size classes,
        because a controller reports ``UR3`` for a UR3e. ``_verify_tool_frame`` compares a
        distance against a tolerance, and over 20000 random joint vectors a ur3/ur3e swap lands
        under the default 10 mm tolerance in 12.2 % of poses, with a minimum separation of
        8.50 mm. So the two arms this cell is most likely to confuse are the two arms neither
        check can separate.

        This one is relative and therefore immune to the tolerance: the flange -> TCP transform
        is one physical thing, and deriving it through the wrong DH table moves it. Whichever
        model puts it closer to what this cell expects is the arm on the other end. A correct
        cell reads about 0 mm for itself and 8.5 mm or more for the twin; a swapped one reverses
        that, and the margin between the two is the evidence rather than either distance.

        Silent when the model has no twin (ur16e), when the twin has no DH table, or when the
        two hypotheses are within ``_TWIN_MARGIN_MM`` of each other, which is the honest reading
        of two explanations that fit equally well.
        """
        from .tool_frame import compare_tool_frames, derive_active_tool_frame

        twin = ur_series_twin(self._capabilities.model)
        if twin is None:
            return
        twin_observed = derive_active_tool_frame(twin, joints, controller_tcp)
        if twin_observed is None:
            return
        twin_d_t, _ = compare_tool_frames(twin_observed, expected)
        if twin_d_t + self._TWIN_MARGIN_MM >= own_d_t:
            return
        raise RobotConnectionError(
            f"ur.model is {self._capabilities.model!r}, and this controller is better "
            f"explained by a {twin!r}: deriving the active tool frame through the "
            f"{self._capabilities.model} DH table lands {own_d_t:.1f} mm from what this cell "
            f"expects, and through the {twin} table only {twin_d_t:.1f} mm. Those two arms "
            f"report the SAME model string to a controller dashboard, so nothing upstream can "
            f"tell them apart, and their flanges can sit as little as 8.5 mm apart, which fits "
            f"inside the tool-frame tolerance. ur.model keys the safety DH chain, the collision "
            f"mesh bundle and the cuRobo robot config, so a cell running the wrong one of this "
            f"pair plans every pose against another robot. Point ur.model at the arm that is "
            f"actually plugged in, or, if it really is a {self._capabilities.model}, check the "
            f"pendant TCP first, because a wrong tool frame can also produce this."
        )

    def connect(self) -> None:
        """Open the RTDE connection to the robot controller.

        Once the connection is up, the configured payload envelope is pushed to the
        controller through ``setPayload``, so the protective-stop calculations of the
        controller reflect the same mass and centre of gravity this stack enforces. The
        push is gated on ``config.safety.payload.enforce``: with the payload guard
        disabled, the driver does not touch the controller payload either.

        It fails closed on a contradictory payload config before opening the socket, on
        two counts. ``enforce: true`` with ``mass_kg: 0.0``, which is the shipped
        default, would push ``setPayload(0.0)`` and overwrite the controller payload for
        a mounted tool silently. ``mass_kg > 0`` with the default ``cog_mm: [0, 0, 0]``
        would declare that tool a point mass at the flange face. Both fail open in the
        dangerous direction, and both corrupt the protective-stop model of the
        controller and its gravity compensation, which is what ``get_tcp_wrench`` reads.
        A bare flange is expressed by ``enforce: false``, which never touches the
        controller payload.
        """
        payload = self.config.safety.payload
        if payload.enforce and payload.mass_kg == 0.0:
            raise RobotConnectionError(
                "safety.payload has enforce: true but mass_kg: 0.0. Connecting would push "
                "setPayload(0.0 kg) and overwrite the controller's configured payload for a mounted "
                "tool, so its protective-stop model would under-read the real mass. "
                "WEIGH THE WHOLE ASSEMBLY, not the gripper: the coupling/adapter plate, every cable and "
                "hose that rides on the wrist, and any workpiece the arm carries. A gripper's datasheet "
                "mass is not this number and no value is shipped here on purpose; a plausible default "
                "is how a cell ends up telling the controller about a tool it is not carrying. Set "
                "safety.payload.mass_kg and cog_mm from the scale, or set safety.payload.enforce: false "
                "if the flange is genuinely bare."
            )
        # The second half of the check above. A declared mass with the default centre of
        # gravity tells the controller a multi-kilogram tool is a point mass at the
        # flange face, which is physically impossible for anything bolted on. Measured:
        # at the UR3e shipped max_mass_kg of 3.0 in robot.ur3e.yaml, and the 132 mm tool
        # this project ships, that is 3.0 * 9.81 * 0.132 = 3.88 Nm of wrist torque the
        # controller does not know about. The same model backs gravity compensation, so
        # it also corrupts get_tcp_wrench(), which this driver documents as the hand-over
        # signal. cog_mm is the same flange-frame bench measurement as the tool frame, so
        # taking one without the other is the gap rather than the effort.
        if payload.enforce and payload.mass_kg > 0.0 and all(v == 0.0 for v in payload.cog_mm):
            raise RobotConnectionError(
                f"safety.payload declares mass_kg: {payload.mass_kg} but leaves cog_mm at its "
                "[0, 0, 0] default; that is the 'not measured yet' marker, and pushing it would "
                "tell the controller the tool is a point mass at the flange face. Its protective-stop "
                "model and its gravity compensation (and therefore get_tcp_wrench) would both be "
                "computed against a tool that cannot exist. Measure the CoG offset from the flange in "
                "mm, the same bench step that gives you the tool frame, and set "
                "safety.payload.cog_mm, or set safety.payload.enforce: false for a genuinely bare "
                "flange. A deliberately axis-centred tool is still not [0, 0, 0]: state its real "
                "stand-off, e.g. [0, 0, 60]."
            )
        # The tool frame is declared before a real arm is allowed to move. Silence here
        # is not a default but an unmeasured cell: every pose this stack commands is the
        # grasp centre, and with the UR factory tool register, which is the identity at
        # the flange, a top-down grasp at z=37 mm drives the flange there and the
        # fingertips about 95 mm below the table. Nothing downstream catches that,
        # because the WorkspaceGuard bounds the commanded number and no profile declares
        # a table fixture.
        tool = self.config.gripper.tool_frame
        if tool.source == "undeclared":
            raise RobotConnectionError(
                "gripper.tool_frame.source is 'undeclared': nobody has said where this gripper's "
                "grasp centre sits on the flange, so the driver cannot know whether the poses it "
                "commands mean the TCP or the flange. Measure the coupling + gripper (the same bench "
                "step as safety.payload.cog_mm) and set gripper.tool_frame.offset_mm + "
                "rotation_quat_xyzw, then choose source: 'willy' (this driver composes the transform "
                "and the controller runs a bare flange) or 'polyscope' (an operator set the TCP on the "
                "teach pendant and the driver only verifies it)."
            )
        self._conn.connect()
        if payload.enforce:
            try:
                self._conn.set_payload(payload.mass_kg, payload.cog_mm)
            except BaseException as exc:  # noqa: BLE001 (the rollback below is the whole point)
                # Roll the connection back: an unverified payload on a connected
                # controller is more dangerous than no connection at all.
                #
                # That invariant is why the catch is unconditional. Naming
                # (RuntimeError, OSError, ValueError) reads as thorough and is not.
                # An ``AttributeError`` from a build that lacks or renames
                # ``setPayload``, a ``TypeError`` from a signature change such as a
                # positional centre of gravity against a list, any vendor-specific
                # exception, and a ``KeyboardInterrupt`` mid-push would all skip the
                # rollback and leave the arm connected with a payload the
                # protective-stop model of the controller does not reflect.
                #
                # Measured on the pinned ur_rtde==1.6.5: the symbol exists, as
                # ``setPayload(mass: float, cog: list[float]) -> bool``. So the first two
                # of those are the guard for the next build rather than a suspicion about
                # this one. A handler whose only job is to fail closed does not enumerate
                # failure types.
                self.logger.error(
                    "UR setPayload failed; rolling back the connection: %s",
                    exc,
                )
                self._conn.disconnect()
                if isinstance(exc, Exception):
                    raise RobotConnectionError(
                        f"UR setPayload({payload.mass_kg} kg) failed: {exc}"
                    ) from exc
                raise  # KeyboardInterrupt and SystemExit propagate unchanged, but disconnected first
        # Last, because it needs a live connection: confirm the controller actual tool
        # frame is the one this config assumes. It fails closed.
        #
        # The rollback lives here rather than inside the check, so every failure path
        # rolls back rather than only the ones that remember to. A branch that raises
        # without disconnecting, such as an unbundled DH table for the model, leaves a
        # live RTDE control script owning the arm with the payload already pushed, while
        # the caller is told the connection failed. A long-running caller then cannot
        # recover: `URConnection.connect()` early-returns on an already-open connection,
        # so the retry skips the socket, re-pushes the payload and fails the same way
        # forever. A one-shot CLI hides that behind process exit.
        try:
            # Before the tool-frame check, because where both would fire this one names
            # the cause and that one only names a symptom. Measured: a UR3e container
            # driven with the default `ur.model: ur5e` is refused with a report that the
            # controller is running a tool frame 177.4 mm away, which sends the reader to
            # the pendant TCP. The real answer, that the DH table the check derives from
            # belongs to a different robot, is one dashboard call away.
            self._verify_controller_model()
            self._verify_tool_frame()
        except BaseException:
            self._conn.disconnect()
            raise

    def start_planner(self) -> "SidecarIdentity":
        """Start this arm's cuRobo planner through the path its first planned move takes, and say what it loaded.

        No controller is asked: the planner is built from this arm's config, so a cell can learn at a desk
        whether its planner starts. Every refusal the first move would meet is raised here, as a
        ``CuroboUnavailableError`` naming it: an undeclared planner margin, no hand, an unplaceable hand, a
        missing retract row, a descriptor built for another robot, and a combination no committed evidence
        file measured. A cell whose ``motion_planner`` is not ``curobo`` starts no planner and is refused
        saying so.
        """
        if self._motion_planner != "curobo":
            raise CuroboUnavailableError(
                f"robot.ur.motion_planner is {self._motion_planner!r}, so this cell starts no planner: its moves are "
                f"solved by the controller's IK and judged by the exact mesh guard alone"
            )
        return self._curobo_ur_planner().start()

    def stop_planner(self) -> None:
        """Shut down the planner :meth:`start_planner` or a move started. Idempotent, and the connection is untouched."""
        if self._curobo_ur is not None:
            self._curobo_ur.close()
            self._curobo_ur = None

    def disconnect(self) -> None:
        """Close the RTDE connection, and shut down the cuRobo planning server where one was started."""
        if self._curobo_ur is not None:
            self._curobo_ur.close()
            self._curobo_ur = None
        self._observed_tool_frame = None
        self._conn.disconnect()

    def __enter__(self) -> URRobotArm:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._conn.is_connected

    @property
    def pose(self) -> URPose:
        """The controller's TCP register (mm, axis-angle rad): the bare flange in ``willy`` mode.

        The grasp centre every Protocol caller speaks is :meth:`get_tcp_pose`.
        """
        return self._motion.get_current_pose()

    @property
    def joints(self) -> list[float]:
        """Current joint angles in radians."""
        return self._conn.get_joint_positions()

    @property
    def tcp_raw(self) -> list[float]:
        """Raw UR TCP ``[x_m, y_m, z_m, rx, ry, rz]``."""
        return self._conn.get_tcp_pose()

    def _gate_pose(self, pose: Pose, command: MotionCommand) -> "MotionResult | None":
        """The full Cartesian pipeline for one commanded pose. ``None`` means every guard passed.

        It is shared with :meth:`move`, so the bool-returning surfaces run the same
        pipeline rather than the workspace box alone. IK is pre-resolved for the same
        reason it is there: without ``target_joints`` the joint-limit, IK-quality and
        arm-against-arm guards have nothing to read and fail closed on every Cartesian
        move. A hand-guided arm is refused first.
        """
        refused = self._hand_guided_refusal(command, target_pose=pose)
        if refused is not None:
            return refused
        if self._preflight is None:
            return None
        target_joints: JointPositions | None = None
        current_joints: JointPositions | None = None
        if self._conn.is_connected:
            try:
                target_joints = self.ik(pose)
            except (RobotKinematicsError, RobotConnectionError) as exc:
                status = (MotionStatus.IK_FAILED if isinstance(exc, RobotKinematicsError)
                          else MotionStatus.CONNECTION_ERROR)
                return MotionResult.failed(
                    status, command, target_pose=pose, message=str(exc), exception=exc,
                )
            try:
                current_joints = self.get_joint_positions()
            except RobotConnectionError:
                current_joints = None  # the IK-jump check skips rather than fails closed
        try:
            ctx = self._preflight.context_for_pose(
                pose, command=command, target_joints=target_joints,
                current_joints=current_joints, arm=self,
            )
            decision = self._preflight.evaluate(ctx)
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_pose=pose,
                message=str(exc), exception=exc,
            )
        return SafetyPreflight.as_motion_result(decision, command, target_pose=pose)

    def _drive_pose(
        self,
        pose: Pose | URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Command a Cartesian pose with no safety gate. Callers must have gated it first."""
        return self._motion.move_to(
            self._to_controller_urpose(pose),
            linear=linear, vel=vel, acc=acc, register=register,
            workspace_pose=self._coerce_urpose(pose),
        )

    def move_to(
        self,
        pose: Pose | URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Move to ``pose``, gated through the full preflight pipeline.

        It accepts a vendor-neutral :class:`Pose`, which the pipelines and the API use,
        or a :class:`URPose`, which is the controller's own frame. Only tests send one:
        the calibration routine refuses anything but a Pose.

        It runs the whole pipeline and not the workspace box alone, which is all
        ``MotionController.move_to`` checks. A surface that ran the box alone would
        command motion no joint-limit, IK-quality, self-collision, fixture, payload or
        continuity guard ever saw, on a cell where :meth:`move` runs all six. ``False``
        therefore also means a guard refused, and the typed reason is logged.

        On a cuRobo cell it goes through :meth:`move`, which is what makes the two surfaces
        the same motion: the planner plans it, or the line is sampled and judged when
        ``linear`` is set. Reaching the controller directly from here would leave a cell
        configured for cuRobo with one verb that does not use it.
        """
        target = self._pose_from_any(pose)
        if self._motion_planner == "curobo":
            result = self.move(target, linear=linear, vel=vel, acc=acc, register=register)
            if not result.ok:
                self.logger.error("move_to REFUSED by %s: %s", result.status, result.message)
            return result.ok
        rejected = self._gate_pose(target, MotionCommand.MOVE_TO)
        if rejected is not None:
            self.logger.error("move_to REFUSED by %s: %s", rejected.status, rejected.message)
            return False
        return self._drive_pose(pose, linear=linear, vel=vel, acc=acc, register=register)

    def _pose_from_any(self, pose: Pose | URPose) -> Pose:
        """A BASE-frame :class:`Pose` from either accepted input, for the guards to read."""
        if isinstance(pose, URPose):
            return self._pose_from_controller(pose, label=pose.label or "move_to")
        return pose

    def is_inside_workspace(self, pose: Pose | URPose) -> bool:
        """The vendor-neutral workspace pre-check the pipelines use.

        It accepts a :class:`Pose` in the BASE frame, or a :class:`URPose`.
        """
        return self._guard.is_inside_workspace(self._coerce_urpose(pose))

    @staticmethod
    def _coerce_urpose(pose: Pose | URPose) -> URPose:
        """A :class:`Pose` as a :class:`URPose` with no tool frame applied.

        A URPose passes through. This is for the workspace box, which bounds the
        commanded TCP, both in :meth:`is_inside_workspace` and as the ``workspace_pose``
        of every controller command. It is not itself a controller command: in ``willy``
        mode the controller runs a bare flange and needs :meth:`_to_controller_urpose`.
        """
        if isinstance(pose, URPose):
            return pose
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"URRobotArm requires Frame.BASE; got {pose.frame!r}."
            )
        return pose_to_urpose(pose)

    def _to_controller_urpose(self, pose: Pose | URPose) -> URPose:
        """What the controller is told for ``pose``.

        A :class:`URPose` is already in the controller frame, which is how
        :meth:`_pose_from_any` reads it, so it passes through. A
        :class:`Pose` is the grasp centre every Protocol caller speaks, and in ``willy``
        mode it becomes the flange target before the controller sees it, the same
        conversion :meth:`ik` and :meth:`move_linear` make. The guard judges the joints
        for that flange target, so every command that reaches the controller carries the
        same target; a grasp centre sent as a flange target would land one tool length
        further along the approach.
        """
        if isinstance(pose, URPose):
            return pose
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"URRobotArm requires Frame.BASE; got {pose.frame!r}."
            )
        return self._pose_to_controller(pose)

    def move_home(self) -> bool:
        """Move to the home joint configuration, gated. ``False`` means refused; :meth:`move_to_home` says why.

        It is ``self.move_to_home().ok``, so every caller that reads a bool keeps its answer, and
        the typed refusal is in the driver's log as ``move_home REFUSED by <status>: <sentence>``.
        """
        return self.move_to_home().ok

    def move_to_home(self) -> MotionResult:
        """Move to the home joint configuration, gated, and say what the gates found (``HomesTyped``).

        This is the natural first motion on a new cell.
        ``MotionController.move_home`` checks the workspace box, logs a warning and
        proceeds anyway, so routing straight to it would leave the one commanded motion
        on the real path that no safety guard ever saw. The measured consequence on a
        UR3e: the UR5e-authored default puts the grasp centre at z = 561.9 mm against
        that cell ``workspace_limits`` z_max of 320, at 93.4% of its reach.

        So it runs the same destination guards as every other joint move, which is
        ``gate_joint_target`` with joint limits, self-collision and payload, and fails
        closed on the workspace check instead of warning past it. Both halves are
        needed and neither subsumes the other: ``gate_joint_target`` deliberately skips
        the Cartesian workspace box, because that guard does not apply to a
        point-to-point joint command, and here the box matters because the home pose is
        a place the arm will sit.

        On a cuRobo cell the whole way to the home configuration is judged as well, by
        :meth:`_judge_joint_move`: the straight line where it is clear, a cuRobo plan around it
        where it collides. The ``moveJ`` is sent here rather than through
        ``MotionController.move_home``: that method sends an unclamped command at the
        controller's own defaults, and a move that was judged has to be the move that runs. The
        configuration the judge returns is the one handed to the drive, so the configuration
        judged is the configuration sent.

        The result is the refusing gate's own: the destination guard's status relabelled
        ``MOVE_HOME``, ``WORKSPACE_REJECTED`` for a home grasp centre outside the box,
        ``CONTROLLER_REJECTED`` for a home the controller's FK could not place, the path
        judge's status, and for the drive ``CONNECTION_ERROR``, ``INVALID_TARGET`` or
        ``CONTROLLER_REJECTED``, whose message says whether ``moveJ`` had been sent. It is
        stamped as :meth:`move_to_joints` stamps its result. With neither a camera world nor a
        decline in scope on a cuRobo cell it is refused first. Every refusal is logged.
        """
        stamp = self._move_camera_world(UNSET)
        refused = self._refused_for_camera_world(stamp, MotionCommand.MOVE_HOME)
        if refused is not None:
            self.logger.error("move_home REFUSED by %s: %s", refused.status, refused.message)
            return refused
        joints = JointPositions(np.asarray(self._home_joints, dtype=np.float64))
        before = self._planner_refresh()
        unstamped = self._move_to_home_unstamped(joints)
        after = self._planner_refresh()
        vouched = after.camera_world() if after is not None and after is not before else None
        result = stamp_result(unstamped, self._move_camera_world(UNSET, planned=vouched))
        if not result.ok:
            self.logger.error("move_home REFUSED by %s: %s", result.status, result.message)
        return result

    def _move_to_home_unstamped(self, joints: JointPositions) -> MotionResult:
        """The body of :meth:`move_to_home`. ``joints`` is the one home every step below reads."""
        refused = self._home_refusal(joints)
        if refused is not None:
            return refused
        target, route, refused = self._judge_joint_move(joints, command=MotionCommand.MOVE_HOME)
        if refused is not None:
            return refused
        return self._drive_judged_joints(target, route, command=MotionCommand.MOVE_HOME, done="move_home")

    def _home_is_admissible(self, verb: str) -> bool:
        """Whether the home passes :meth:`_home_refusal`, logged under ``verb`` when it does not."""
        refused = self._home_refusal(JointPositions(np.asarray(self._home_joints, dtype=np.float64)))
        if refused is not None:
            self.logger.error("%s REFUSED by %s: %s", verb, refused.status, refused.message)
        return refused is None

    def _home_refusal(self, joints: JointPositions) -> MotionResult | None:
        """The home gate on ``joints``: the typed refusal, or ``None`` where the move may be judged as a path.

        The destination guards run first, then the workspace box on the home pose. The box
        bounds the grasp centre, so the controller FK is converted out of its TCP register
        first: in ``willy`` mode that register is the bare flange, and boxing it would pass a
        home whose grasp centre lies a tool length outside. An FK that cannot be read refuses,
        because an unchecked home is what this gate stops. An arm that is not connected has no
        FK to ask; the path judge and the drive refuse it after this.
        """
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                message = rejected.message
                if float(np.max(np.abs(joints.values))) > 2.0 * np.pi:
                    # Every UR joint stops at one full turn either way, so a value beyond 2*pi cannot be
                    # reached in radians, and a home copied off the pendant in degrees is how one is written.
                    message += (" robot.home_joint_positions holds a value beyond 2*pi, which reads as degrees; "
                                "the key is in radians.")
                return dataclasses.replace(
                    rejected, command=MotionCommand.MOVE_HOME, target_joints=joints, message=message,
                )
        if not self._conn.is_connected:
            return None
        try:
            reported = URPose.from_ur_list(self._conn.fk(joints.tolist()), label="home")
            home_pose = self._coerce_urpose(self._pose_from_controller(reported, label="home"))
            inside = self._guard.is_inside_workspace(home_pose)
        except Exception as exc:  # noqa: BLE001 (FK unavailable must not silently wave the move through)
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_HOME, target_joints=joints,
                message=(
                    f"could not check the home pose: the controller's forward kinematics of the home "
                    f"configuration could not be read ({type(exc).__name__}: {exc}), and a home the "
                    f"workspace box never saw is not sent"
                ),
                exception=exc,
            )
        if inside:
            return None
        return MotionResult.failed(
            MotionStatus.WORKSPACE_REJECTED, MotionCommand.MOVE_HOME, target_joints=joints,
            message=self._home_outside_the_box(home_pose),
        )

    def _home_outside_the_box(self, home_pose: URPose) -> str:
        """Why a home grasp centre outside ``workspace_limits`` is refused: the box, and what it bounds."""
        box = self._guard.limits
        boxed = {
            "willy": ("The box bounds the grasp centre, the flange the controller's FK reports plus the declared "
                      "tool frame, and not the flange itself."),
            "polyscope": "The box bounds the grasp centre, the TCP the controller's FK reports with the pendant's tool.",
        }.get(self.config.gripper.tool_frame.source,
              "The box bounds the grasp centre, and with no tool frame declared that is the flange the FK reports.")
        if tuple(float(v) for v in self._home_joints) == HOME_JOINTS_DEFAULT:
            fix = ("No robot.home_joint_positions is set, so this is the shipped default, the arm standing "
                   "straight up; set one whose grasp centre lies inside this cell's box.")
        else:
            fix = ("Set robot.home_joint_positions, in radians, to a configuration whose grasp centre lies "
                   "inside this cell's box.")
        return (
            f"the home grasp centre ({home_pose.x:.1f}, {home_pose.y:.1f}, {home_pose.z:.1f}) mm is outside "
            f"workspace_limits (x {box.x_min:.1f} to {box.x_max:.1f}, y {box.y_min:.1f} to {box.y_max:.1f}, "
            f"z {box.z_min:.1f} to {box.z_max:.1f} mm). {boxed} {fix} Widening the box widens it for every "
            f"move, not only for home."
        )

    def stop(self) -> None:
        """Emergency stop."""
        self._conn.stop()

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Delegate to the underlying RTDE :meth:`URConnection.wait_until_steady`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("wait_until_steady() requires an open connection.")
        return bool(self._conn.wait_until_steady(timeout_s, poll_interval_s))

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
        camera_world: Maybe[CameraWorldDecline] = UNSET,
    ) -> MotionResult:
        """The typed counterpart of :meth:`move_to`.

        Every commanded pose goes through the vendor-neutral :class:`SafetyPreflight`
        pipeline. A rejection from any guard produces the matching typed
        :class:`MotionStatus`, one of ``WORKSPACE_REJECTED``, ``JOINT_LIMIT_REJECTED``,
        ``IK_QUALITY_REJECTED``, ``SELF_COLLISION_REJECTED``, ``PAYLOAD_REJECTED`` and
        ``CONTINUITY_REJECTED``, so a caller branches on the precise cause without
        scraping a log. A connection fault raised by the underlying bool path is caught
        and reported as :attr:`MotionStatus.CONNECTION_ERROR`.

        The result says what stood behind the motion: UNPLANNED on the ik planner, and on cuRobo
        DECLINED for a decline (``camera_world``, or :meth:`without_camera_world`), MISSING while
        no live camera world is wired, and once one is, PLANNED when the refresh this motion made
        vouched for the cell and UNSTATED when none did, which is why it is read after the motion. A
        declined motion on an arm whose live camera world is wired is refused with ``UNSUPPORTED``
        before the planner is asked, because the planner is handed that world before every plan and
        cannot set it aside for one motion. A MISSING motion, with neither a world nor a decline, is
        refused the same way (``NO_CAMERA_WORLD_MESSAGE``), before it reads a connection or starts a
        planner.
        """
        stamp = self._move_camera_world(camera_world)
        refused = self._refused_for_camera_world(stamp, MotionCommand.MOVE_TO, target_pose=pose)
        if refused is not None:
            return refused
        before = self._planner_refresh()
        unstamped = self._move_unstamped(pose, linear=linear, vel=vel, acc=acc, register=register)
        after = self._planner_refresh()
        vouched = after.camera_world() if after is not None and after is not before else None
        return stamp_result(unstamped, self._move_camera_world(camera_world, planned=vouched))

    def _refused_for_camera_world(
        self, stamp: CameraWorldStamp, command: MotionCommand, *,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> MotionResult | None:
        """The ``UNSUPPORTED`` result a motion stamped ``stamp`` is refused with before its body, or ``None``.

        One rule for every verb (``core.camera_world.camera_world_refusal``): MISSING is refused, and
        DECLINED is refused where a live world is wired. Every motion verb asks this before its body, so a
        hand-guided arm is refused here first (:meth:`_hand_guided_refusal`), whatever the stamp says.
        """
        refused = self._hand_guided_refusal(command, target_pose=target_pose, target_joints=target_joints)
        if refused is not None:
            return refused
        message = camera_world_refusal(stamp, live_world_wired=self._live_world is not None)
        if message is None:
            return None
        return MotionResult.failed(
            MotionStatus.UNSUPPORTED, command, target_pose=target_pose, target_joints=target_joints,
            message=message, camera_world=stamp,
        )

    def _move_camera_world(
        self, keyword: Maybe[CameraWorldDecline], *, planned: CameraWorldStamp | None = None
    ) -> CameraWorldStamp:
        """What stands behind a ``move`` on this arm, read off the arm as it is now, never off config.

        ``planned`` is the stamp of the refresh the motion made, which only the motion can know.
        """
        curobo = self._motion_planner == "curobo"
        return resolve_camera_world(
            unplanned=None if curobo else f"robot.ur.motion_planner is {self._motion_planner!r}",
            missing=None if self._live_world is not None else _UR_NO_LIVE_WORLD,
            keyword=keyword,
            block=active_decline(self),
            planned=planned,
        )

    def _planner_refresh(self) -> "WorldRefresh | None":
        """The planner's most recent world refresh, or ``None`` before it made one or for a double.

        Compared by identity before and after a motion: a refresh that is the same object as before
        was made for an earlier motion, and must not vouch for this one.
        """
        from src.robot.safety.planning import live_world as _live

        refresh = getattr(self._curobo_ur, "last_world_refresh", None) if self._curobo_ur is not None else None
        return refresh if isinstance(refresh, _live.WorldRefresh) else None

    def _move_unstamped(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """The body of :meth:`move`. It stamps nothing; the public verb does."""
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"URRobotArm.move requires Frame.BASE; got {pose.frame!r}",
            )
        if self._motion_planner == "curobo":
            # Dropping `linear` here would make `move(pose, linear=True)` on a cuRobo cell
            # plan a path instead of running the line the caller asked for. It is a
            # different motion: the planner routes around obstacles and the controller
            # does not.
            if linear:
                return self._drive_checked_line(pose, vel=vel, acc=acc)
            return self._drive_curobo(pose, vel=vel, acc=acc)
        # Pre-resolve IK on the driver side, so the preflight pipeline sees
        # ``target_joints`` for every Cartesian command. Joint-limit, IK-quality and the
        # arm-against-arm part of the self-collision guard would otherwise fail closed as
        # unavailable on every Cartesian move.
        target_joints: JointPositions | None = None
        current_joints: JointPositions | None = None
        if self._conn.is_connected:
            try:
                target_joints = self.ik(pose)
            except RobotKinematicsError as exc:
                return MotionResult.failed(
                    MotionStatus.IK_FAILED,
                    MotionCommand.MOVE_TO,
                    target_pose=pose,
                    message=str(exc),
                    exception=exc,
                )
            except RobotConnectionError as exc:
                return MotionResult.failed(
                    MotionStatus.CONNECTION_ERROR,
                    MotionCommand.MOVE_TO,
                    target_pose=pose,
                    message=str(exc),
                    exception=exc,
                )
            try:
                current_joints = self.get_joint_positions()
            except RobotConnectionError:
                # A telemetry hiccup mid-call. The IK-jump check skips rather than
                # failing closed, because ``current_joints`` is None.
                current_joints = None
        try:
            ctx = self._preflight.context_for_pose(
                pose,
                command=MotionCommand.MOVE_TO,
                target_joints=target_joints,
                current_joints=current_joints,
                arm=self,
            )
            decision = self._preflight.evaluate(ctx)
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        rejected = SafetyPreflight.as_motion_result(
            decision, MotionCommand.MOVE_TO, target_pose=pose,
        )
        if rejected is not None:
            return rejected
        try:
            # The ungated primitive. The pipeline above has already judged this pose.
            ok = self._drive_pose(
                pose, linear=linear, vel=vel, acc=acc, register=register,
            )
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        # Attach the real rejection cause the inner controller classified, such as a
        # genuine near-singularity as IK_QUALITY_REJECTED, rather than the generic
        # CONTROLLER_REJECTED default of from_bool.
        failure_status = (
            self._motion.last_reject_status
            if self._motion.last_reject_status is not None
            else MotionStatus.CONTROLLER_REJECTED
        )
        return MotionResult.from_bool(
            ok, MotionCommand.MOVE_TO, target_pose=pose, failure_status=failure_status,
        )

    def _drive_curobo(
        self,
        pose: Pose,
        *,
        vel: float | None = None,
        acc: float | None = None,
    ) -> MotionResult:
        """Take the flange to ``pose`` on the route the arm chooses, judge the legs that will run, and run exactly those.

        In order, and nothing moves before the last of the judging has passed (:meth:`_route_to_the_nearest_goal`):

        1. The goal configuration. Every closed-form inverse kinematics solution of the flange goal is turned onto
           the full turns nearest the arm inside the planner's joint window and the joint-limit guard's (the cable
           window about home where the cell keeps one), and ranked: the branch the arm holds first, then the least
           time a ``moveJ`` there takes. Each is screened by the planner at that one configuration and by the endpoint
           gate. cuRobo never chooses a goal configuration.
        2. The straight joint line to each admissible goal, in rank order, judged by the local path gate and by the
           planner against its world, the camera's included, at ``safety.planned_motion.line_clearance_mm``. The
           first that passes runs as it is, one leg.
        3. Only where no line passes: cuRobo plans to the goals in the same order, up to
           ``_NEAREST_GOALS_PLANNED`` of them. A plan is taken where it ends on the configuration asked for and no
           joint swings more than ``safety.planned_motion.max_detour_deg`` beyond its span; its end then meets the
           controller's kinematics and the endpoint gate, and its legs, shortened to the fewest of its own waypoints,
           meet both authorities. A branch other than the arm's is tried only after every goal on it failed, and
           taking one is a warning naming both.

        Exactly the list that passed runs, one ``moveJ`` per waypoint, at the speed clamped to ``motion_limits``, and
        one log line says how it was chosen and how far each joint turns on it.

        It is fail-closed. An unavailable cuRobo gives ``CONTROLLER_REJECTED``; a goal out of reach ``IK_FAILED``; a
        goal with no configuration inside the window ``JOINT_LIMIT_REJECTED``; a planner that will not start from
        where the arm stands the status of what it found there; a goal every configuration of which the endpoint gate
        refuses, that gate's status; no clear line and only plans past the detour bound ``JOINT_LIMIT_REJECTED``
        naming the joint; no line and no plan ``TIMEOUT`` with the no-plan sentence, the refusal a pick may look again
        at, and the log line naming every goal and why it failed; a guard or the planner refusing a plan's end or legs
        the matching status. There is no blind motion.
        """
        if not self._conn.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, target_pose=pose,
                message="cuRobo move requires an open connection.",
            )
        planner = self._curobo_ur_planner()
        # cuRobo plans to tool0, so it gets the flange goal in both ownership modes. Its
        # model has no notion of the controller tool register, because it never talks to
        # the controller at all, and its 2F-85 collision spheres are attached to tool0,
        # so a grasp-centre goal would both drive the flange to the grasp point and place
        # the gripper collision geometry a whole tool length past the target. The pose
        # reported back to the caller stays the TCP pose they asked for.
        goal = self._pose_to_flange(pose)
        # Clamped once, here: the same speed ranks the goals and runs the moveJ.
        vel, acc = self._motion._clamp(vel, acc)
        # Ranking reads the speed, but a speed that is no speed is ur_rtde's to refuse at the moveJ, not a reason to
        # plan differently: one speed for every joint orders the goals the same whatever it is.
        ranked_at = (float(vel) if vel is not None and math.isfinite(float(vel)) and float(vel) > 0.0
                     else float(self.config.motion_limits.max_velocity))
        try:
            route = self._route_to_the_nearest_goal(planner, goal, pose, velocity=ranked_at)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        if isinstance(route, MotionResult):
            return route
        self._log_the_route(f"move to {pose.label or '<unlabeled>'}", route)
        return planner.execute(route.waypoints, pose, vel=vel, acc=acc)

    #: How many configurations of one goal cuRobo is asked to plan to, once no straight line to any of them is clear. A
    #: joint plan that fails costs cuRobo all its attempts, about 7 to 9 s a goal measured on the UR10 descriptor on this
    #: project's RTX 5080 (2026-09-23), so the count is small: a goal no plan reaches pays up to three of those.
    _NEAREST_GOALS_PLANNED = 3
    #: How close, on every joint, a joint plan has to end to the configuration it was asked for to be taken. cuRobo
    #: ends a joint plan on its goal because its optimiser works locally (0.0 rad measured on the probed legs), and
    #: its own documentation does not promise it, so it is read rather than assumed.
    _NEAREST_GOAL_END_TOL_RAD = 1e-4

    def _route_to_the_nearest_goal(
        self, planner: CuroboUrPlanner, goal: Pose, pose: Pose, *, velocity: float,
    ) -> "_Route | MotionResult":
        """The judged route to the configuration of ``goal`` nearest the arm, or the typed refusal of the move.

        ``goal`` is the flange goal and ``pose`` the TCP pose the caller asked for, which every refusal names. Raises
        ``CuroboUnavailableError`` where the planner cannot be reached.

        cuRobo, handed a pose, picks a configuration for it from random seeds over the whole joint range and does not
        prefer the one nearest the arm: measured on a UR10, a wrist sweep took 8 of 22 legs the long way, and the same
        leg planned twice took different ways. So cuRobo is never handed a pose. Every closed-form solution of the
        flange goal is turned onto the full turns nearest the joints the arm stands at, inside the window both the
        planner and the joint-limit guard admit, and ranked by the branch first and the least time a ``moveJ`` there
        takes after it. The goals on the arm's branch are screened, lined and planned to first; the others only after
        all of those failed. The world is refreshed once, here, and every screen, line and plan is judged in it.
        """
        try:
            here = [float(v) for v in self._conn.get_joint_positions()]
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"a planned move starts at the current configuration, and the controller's joint "
                         f"positions could not be read ({type(exc).__name__}: {exc}); nothing was sent"),
                exception=exc,
            )
        model = str(self.config.ur.model)
        solutions = (ur_flange_ik(model, np.asarray(goal.to_matrix(), dtype=np.float64), q6_if_singular=here[-1])
                     if len(here) == self.capabilities.dof else None)
        if solutions is None:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"robot.ur.model {model!r} has no DH chain to solve this goal on, or the controller reported "
                         f"{len(here)} joints, so no configuration of it can be chosen; cuRobo is never left to "
                         "choose one, and nothing was sent"),
            )
        if not solutions:
            return MotionResult.failed(
                MotionStatus.IK_FAILED, MotionCommand.MOVE_TO, target_pose=pose,
                message="the flange goal of this pose is out of the arm's reach: it has no inverse kinematics solution",
            )
        lower, upper = self._goal_joint_window()
        goals = nearest_goals(solutions, current=here, lower=lower, upper=upper, velocity=velocity)
        if not goals:
            half_turn = any(bool(getattr(g, "within_half_turn_of_home", False))
                            for g in (getattr(self._preflight, "guards", ()) or ()))
            return MotionResult.failed(
                MotionStatus.JOINT_LIMIT_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"none of the {len(solutions)} inverse kinematics solution(s) of this goal turns inside the "
                         "planner's and the joint-limit guard's joint window"
                         + (", which keeps half a turn either side of home "
                            "(safety.joint_limits.within_half_turn_of_home)" if half_turn else "")
                         + "; cuRobo is never left to choose a configuration outside it, and nothing was sent"),
            )
        refused = self._refresh_before_the_path_gate(
            near_point_mm=[float(v) for v in goal.position_mm], command=MotionCommand.MOVE_TO,
            target_pose=pose, goal_tcp_mm=self._tcp_of_pose(pose),
        )
        if refused is not None:
            return refused
        held = ur_branch(model, here)
        branches = {g: ur_branch(model, g.joints) for g in goals}
        # The arm's own branch first, in the order nearest_goals gave; a configuration at a branch point agrees with
        # both sides of it, so out of a candle-straight home, all three branch points at once, every goal is on it.
        on_it = [g for g in goals if _same_branch(held, branches[g])]
        groups = (on_it, [g for g in goals if g not in on_it])
        names = {g: (f"goal {rank} of {len(goals)} ({_branch_text(branches[g])}, "
                     f"{math.degrees(g.largest_rad):.1f} deg at most)")
                 for rank, g in enumerate(groups[0] + groups[1], start=1)}
        tried: list[str] = []
        end_refusals: list[MotionResult] = []
        detours: list[tuple[float, str]] = []
        planned = 0
        for group in groups:
            admissible: list[NearestGoal] = []
            for candidate in group:
                verdict = planner.check_joint_path([list(candidate.joints)], refresh=False)
                if not verdict.valid:
                    tried.append(_with_refusal(f"{names[candidate]} refused by the planner",
                                               getattr(verdict, "refusal", None)))
                    continue
                refused_end = self._gate_planned_config(pose, JointPositions(candidate.joints))
                if refused_end is not None:
                    tried.append(f"{names[candidate]} refused by the endpoint gate: {refused_end.status.value}")
                    end_refusals.append(refused_end)
                    continue
                admissible.append(candidate)
            for candidate in admissible:
                line = [list(here), list(candidate.joints)]
                refused = self._judge_legs(planner, line, command=MotionCommand.MOVE_TO, target_pose=pose,
                                           clearance_mm=self._line_clearance_mm())
                if refused is None:
                    return self._route(line, planned=False, how=f"direct line to {names[candidate]}", tried=tried,
                                       held=held, reached=branches[candidate])
                if isinstance(refused.exception, CuroboUnavailableError):
                    return refused  # no line and no plan is judged by a planner that cannot be reached
                tried.append(f"the straight line to {names[candidate]} refused ({refused.status.value}: "
                             f"{refused.message})")
            for candidate in admissible:
                if planned >= self._NEAREST_GOALS_PLANNED:
                    break
                planned += 1
                trajectory, why, start = self._planned(planner, candidate.joints, name=names[candidate])
                if start is not None:
                    return self._start_refusal(start, MotionCommand.MOVE_TO, target_pose=pose)
                if trajectory is None:
                    tried.append(why)
                    continue
                over_deg, detour = self._detour(trajectory)
                if detour:
                    said = f"cuRobo's plan to {names[candidate]} {detour}"
                    tried.append(f"{said}, so it is not taken")
                    detours.append((over_deg, said))
                    continue
                # The plan ends on its goal, or nothing moves. The end check reads where cuRobo's own last
                # configuration puts the flange on the controller's kinematics, which a planner answering in another
                # base misses by a metre; the endpoint gate judges that configuration with the commanded pose in hand,
                # so the box applies as well as the joint limits, self-collision and payload. The continuity guards
                # stay skipped, as in the sim: cuRobo owns a plan's continuity, and a wrap from +pi to -pi is not a
                # blind interpolator teleporting. Cheap, and the plan's legs are judged after.
                refused = (self._plan_end_refusal(goal, trajectory[-1], pose)
                           or self._gate_planned_config(pose, JointPositions(trajectory[-1])))
                if refused is not None:
                    return refused
                waypoints, refused = self._judged_waypoints(
                    planner, trajectory, command=MotionCommand.MOVE_TO, target_pose=pose,
                )
                if refused is not None:
                    return refused
                return self._route(waypoints, planned=True, dense=len(trajectory),
                                   how=f"cuRobo plan to {names[candidate]}", tried=tried, held=held,
                                   reached=branches[candidate])
        self.logger.error("move to %s refused, nothing was sent: %s", pose.label or "<unlabeled>",
                          "; ".join(tried) or "no goal was admissible")
        if len(end_refusals) == len(goals):
            # Every configuration met the endpoint gate and none passed it: the end is the gate's to name, as a plan's
            # end always was, and there was nothing a planner could have been asked.
            first = end_refusals[0]
            return dataclasses.replace(first, message=(
                f"no configuration of this goal passes the endpoint gate; the nearest was refused: {first.message}"))
        if detours:
            # cuRobo found a way, and the only way it found swings a joint around: a refusal that names the joint, not
            # the no-plan sentence a pick answers by looking again at a scene that is not the problem.
            return MotionResult.failed(
                MotionStatus.JOINT_LIMIT_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"no straight line to this goal is clear, and no plan cuRobo found keeps every joint within "
                         f"the detour bound; the furthest: {max(detours)[1]}; nothing was sent"),
            )
        return MotionResult.failed(
            MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose, message=NO_PLAN_FAIL_SAFE_MESSAGE,
        )

    def _planned(
        self, planner: CuroboUrPlanner, goal: "Sequence[float]", *, name: str,
    ) -> "tuple[list[list[float]] | None, str, object | None]":
        """cuRobo's plan to ``goal``, as ``(trajectory, "", None)``, or ``(None, why, None)`` where it is not taken.

        ``(None, "", refusal)`` is a start the planner will not leave, which no other goal plans from either. A plan
        is taken where it ends within ``_NEAREST_GOAL_END_TOL_RAD`` of ``goal``; how far it swings is the caller's to
        read (:meth:`_detour`). Only the goal is ever chosen here; the trajectory is cuRobo's and is never altered.
        Raises ``CuroboUnavailableError``.
        """
        trajectory = planner.plan_joint(list(goal), refresh=False)
        if not trajectory:
            refusal = getattr(planner, "last_refusal", None)
            if refusal is not None and getattr(refusal, "where", None) == StateWhere.START:
                return None, "", refusal
            return None, _with_refusal(f"{name} did not plan", refusal), None
        off_rad = max(abs(float(a) - float(b)) for a, b in zip(trajectory[-1], goal))
        if off_rad > self._NEAREST_GOAL_END_TOL_RAD:
            return None, f"{name} planned to a configuration {off_rad:.2g} rad from it, so the plan is not taken", None
        return trajectory, "", None

    def _start_refusal(
        self, refusal: object, command: MotionCommand, *,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> MotionResult:
        """The typed refusal of a move the planner will not start: the status of what it found where the arm stands.

        A start the planner refuses is not a goal it could not reach, and TIMEOUT with the no-plan sentence would send
        a recovery to look again at a scene that is not the problem.
        """
        return MotionResult.failed(
            _planner_refusal_status(refusal), command, target_pose=target_pose, target_joints=target_joints,
            message=_with_refusal(
                "the arm stands where the planner will not start from, so no plan exists from here and nothing was "
                "sent; bring the arm out of this configuration before it plans again", refusal),
        )

    def _route(
        self, waypoints: "Sequence[Sequence[float]]", *, planned: bool, how: str, dense: int = 2,
        tried: "Sequence[str]" = (), held: "URBranch | None" = None, reached: "URBranch | None" = None,
    ) -> _Route:
        """The route that passed, saying how it was chosen; a change of branch on it is a warning naming both.

        ``planned`` says whether cuRobo planned ``waypoints`` or they are the two ends of a straight line, ``dense``
        how many waypoints cuRobo returned, and ``tried`` what failed before this route, in the order it was tried.
        """
        if tried:
            how += f", after {'; '.join(tried)}"
        if not _same_branch(held, reached):
            self.logger.warning(
                "this move changes the arm's branch from %s to %s: no configuration on the branch it holds could be "
                "reached (%s)", _branch_text(held), _branch_text(reached), "; ".join(tried) or "none was admissible",
            )
            how += f"; branch change from {_branch_text(held)} to {_branch_text(reached)}"
        return _Route(waypoints=waypoints, dense=dense, how=how, planned=planned)

    def _line_clearance_mm(self) -> float:
        """How far a straight joint line has to stay from the planner's world: ``safety.planned_motion``."""
        return float(self.config.safety.planned_motion.line_clearance_mm)

    def _detour(self, waypoints: "Sequence[Sequence[float]]") -> "tuple[float, str]":
        """How far a path swings its worst joint beyond the span between its start and its goal, where that is too far.

        ``(degrees, sentence)`` where some joint swings more than ``safety.planned_motion.max_detour_deg`` past the
        span, the sentence naming the joint; ``(0.0, "")`` where every joint stays within it. The bound is read on the
        waypoints: the legs between them are straight joint lines and add nothing to it
        (:func:`~src.robot.safety.path_samples.span_overshoot_rad`).
        """
        from .curobo_motion import UR_ARM_JOINT_NAMES

        cap_deg = float(self.config.safety.planned_motion.max_detour_deg)
        try:
            overshoot = [math.degrees(v) for v in span_overshoot_rad(waypoints)]
        except ValueError:  # a waypoint that cannot be read: the path gate refuses it and says why
            return 0.0, ""
        if not overshoot or max(overshoot) <= cap_deg:
            return 0.0, ""
        worst = max(range(len(overshoot)), key=overshoot.__getitem__)
        joint = UR_ARM_JOINT_NAMES[worst] if len(overshoot) == len(UR_ARM_JOINT_NAMES) else f"joint {worst + 1}"
        return overshoot[worst], (f"swings {joint} {overshoot[worst]:.1f} deg beyond the span between its start and "
                                  f"its goal, more than safety.planned_motion.max_detour_deg of {cap_deg:g}")

    def _goal_joint_window(self) -> tuple[list[float], list[float]]:
        """Where a chosen goal's joints may lie, in radians: the planner's window, narrowed to what the guard admits.

        The joint-limit guard's table less its margin, the numbers it refuses outside of. Where the guard keeps
        half a turn about home (``safety.joint_limits.within_half_turn_of_home``), that is the table narrowed to
        the window about this arm's home, so no goal is chosen outside it. With no guard, or no table for this
        arm, the planner's window alone; a goal outside what the guard would admit is then the endpoint gate's
        to refuse, as for any plan.
        """
        from .curobo_motion import PLANNER_JOINT_ENVELOPE_RAD

        lower, upper = list(PLANNER_JOINT_ENVELOPE_RAD[0]), list(PLANNER_JOINT_ENVELOPE_RAD[1])
        guard = next((
            candidate for candidate in (getattr(self._preflight, "guards", ()) or ())
            if getattr(candidate, "name", None) == "joint_limit" and callable(getattr(candidate, "limits_for", None))
        ), None)
        if guard is None:
            return lower, upper
        for_arm = getattr(guard, "limits_for_arm", None)
        limits = (for_arm(self) if callable(for_arm)
                  else guard.limits_for(vendor=self.capabilities.vendor, model=self.capabilities.model))
        if limits is None or len(limits[0]) != len(lower):
            return lower, upper
        margin = float(getattr(guard, "margin_deg", 0.0))
        for axis, (low_deg, high_deg) in enumerate(zip(*limits)):
            lower[axis] = max(lower[axis], math.radians(float(low_deg) + margin))
            upper[axis] = min(upper[axis], math.radians(float(high_deg) - margin))
        return lower, upper

    def _judged_waypoints(
        self, planner: CuroboUrPlanner, traj_ur: "list[list[float]]", *, command: MotionCommand,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> "tuple[Sequence[Sequence[float]], MotionResult | None]":
        """The waypoints to run, one ``moveJ`` each, with both authorities having judged their legs; or the refusal.

        cuRobo samples a plan every 25 ms, and a ``moveJ`` starts from rest and stops at its target, so running
        every sample stops the arm at every sample. The plan is shortened first to the fewest of its own
        waypoints whose legs stay within the path gate's step of it, in the gate's own metric
        (:func:`~src.robot.safety.path_samples.simplify_joint_path`); on 12 measured UR10 plans 8 became one
        ``moveJ``. The chords it draws are legs nobody judged, so the shortened list is judged whole, by the local
        path gate and by the planner, before it may run. Where either refuses, the plan as cuRobo returned it is
        judged by both instead, and where that is refused too the move is.

        What is returned is the very list that passed, and the caller hands it to ``execute`` unchanged: the
        legs judged are the legs run. The detour bound is read on it once more, although a subset of a plan's own
        waypoints that keeps both ends cannot swing further than the plan did, so the bound is stated on the list
        that runs. A cell whose preflight samples no path cannot shorten one, and its plan meets
        ``gate_planned_path`` as it always did, which refuses it.
        """
        from src.robot.safety.path_samples import simplify_joint_path

        running: "Sequence[Sequence[float]]" = traj_ur
        step = self._preflight.path_step_mm if self._preflight is not None else None
        reach = self._preflight.joint_radii_mm(self) if step is not None else None
        refused: "MotionResult | None" = None
        if step is not None and reach is not None:
            shortened: "tuple[tuple[float, ...], ...] | None"
            try:
                shortened = simplify_joint_path(traj_ur, reach_mm=reach, tolerance_mm=step)
            except ValueError:  # a waypoint that cannot be read: the path gate below refuses it and says why
                shortened = None
            if shortened is not None and len(shortened) < len(traj_ur):
                refused = self._judge_legs(planner, shortened, command=command, target_pose=target_pose,
                                           target_joints=target_joints)
                if refused is None:
                    running = shortened
                else:
                    self.logger.info(
                        "the plan shortened to %d of its %d waypoints was refused (%s: %s); the plan is judged as "
                        "cuRobo returned it", len(shortened), len(traj_ur), refused.status.value, refused.message,
                    )
        if running is traj_ur:
            refused = self._judge_legs(planner, traj_ur, command=command, target_pose=target_pose,
                                       target_joints=target_joints)
            if refused is not None:
                return (), refused
        _, detour = self._detour(running)
        if detour:
            return (), MotionResult.failed(
                MotionStatus.JOINT_LIMIT_REJECTED, command, target_pose=target_pose, target_joints=target_joints,
                message=f"the path that would run {detour}; nothing was sent",
            )
        return running, None

    def _judge_legs(
        self, planner: CuroboUrPlanner, waypoints: "Sequence[Sequence[float]]", *, command: MotionCommand,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None, clearance_mm: float = 0.0,
    ) -> "MotionResult | None":
        """Both authorities on the legs between neighbouring ``waypoints``; ``None`` means every leg passed.

        The local path gate first, the exact meshes and the declared fixtures, then the planner on the same samples
        against the world it holds, the camera's and the carried part's. The world was refreshed before, so the
        planner is asked with ``refresh=False`` and both judge the same cell; where no step or reach derives,
        ``gate_planned_path`` has already refused and the planner is not asked. ``clearance_mm`` is the world
        clearance the planner holds the samples to, which a straight line the arm would run instead of a plan is asked
        for (``safety.planned_motion.line_clearance_mm``); a plan's legs are judged at none, as cuRobo validated them.
        """
        refused = self._preflight.gate_planned_path(waypoints, arm=self, command=command)
        if refused is not None:
            return refused
        step = self._preflight.path_step_mm if self._preflight is not None else None
        if step is None:  # no self-collision guard: gate_planned_path already refused
            return None
        reach = self._preflight.joint_radii_mm(self)
        if reach is None:
            return None  # gate_planned_path refused this already; here it would be a second voice
        from src.robot.safety.path_samples import waypoint_path_samples

        samples = waypoint_path_samples(waypoints, reach_mm=reach, max_step_mm=step)
        try:
            verdict = planner.check_joint_path(samples.configs, refresh=False, clearance_mm=clearance_mm)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_pose=target_pose, target_joints=target_joints,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        if verdict.valid:
            return None
        refusal = getattr(verdict, "refusal", None)
        what = "this straight joint line" if len(waypoints) == 2 else "the legs this path would run"
        return MotionResult.failed(
            _planner_refusal_status(refusal), command, target_pose=target_pose, target_joints=target_joints,
            message=_with_refusal(f"the planner refused {what}: {verdict.reason}", refusal),
        )

    def _log_the_route(self, what: str, route: _Route) -> None:
        """One line per move: how its route was chosen, how many waypoints ran, and how far each joint turns on them.

        The travel is read on the list that runs, leg by leg: per joint its total turn and its largest single leg, in
        degrees. A long way round shows as a joint turning far more than its end needs, its travel against the
        shortest turn to the same angle; past half a turn more it is a warning, because an unwind the joint limits
        force looks the same and a person should read which it was.
        """
        from .curobo_motion import UR_ARM_JOINT_NAMES

        path = np.asarray([[float(v) for v in w] for w in route.waypoints], dtype=np.float64)
        legs = np.abs(np.diff(path, axis=0)) if len(path) > 1 else np.zeros((1, path.shape[1]))
        travel, largest = legs.sum(axis=0), legs.max(axis=0)
        names = UR_ARM_JOINT_NAMES if len(UR_ARM_JOINT_NAMES) == len(travel) else tuple(
            f"joint {i + 1}" for i in range(len(travel)))
        per_joint = ", ".join(f"{name} {math.degrees(float(t)):.1f}/{math.degrees(float(m)):.1f}"
                              for name, t, m in zip(names, travel, largest))
        self.logger.info(
            "%s: %s; %d waypoint(s) dense, %d executed (%d leg(s)); joint travel in deg, total/largest leg: %s",
            what, route.how, route.dense, len(path), max(len(path) - 1, 0), per_joint,
        )
        shortest = np.abs((path[-1] - path[0] + np.pi) % (2.0 * np.pi) - np.pi)
        extra = travel - shortest
        if float(extra.max()) > math.pi:
            worst = int(np.argmax(extra))
            self.logger.warning(
                "%s turns %s %.1f deg where %.1f deg reaches the same angle: the long way round, or an unwind the joint "
                "limits force", what, names[worst], math.degrees(float(travel[worst])),
                math.degrees(float(shortest[worst])),
            )

    def _gate_planned_config(
        self, pose: Pose, final_joints: JointPositions
    ) -> "MotionResult | None":
        """Gate a planner's final configuration with the box and the joint guards; ``None`` = accepted."""
        if self._preflight is None:
            return None
        current_joints: JointPositions | None = None
        if self._conn.is_connected:
            try:
                current_joints = self.get_joint_positions()
            except RobotConnectionError:
                current_joints = None
        ctx = self._preflight.context_for_pose(
            pose, command=MotionCommand.MOVE_TO, target_joints=final_joints,
            current_joints=current_joints, arm=self,
        )
        decision = self._preflight.evaluate(
            ctx, skip_guards=frozenset({"ik_quality", "motion_continuity"})
        )
        return SafetyPreflight.as_motion_result(
            decision, MotionCommand.MOVE_TO, target_pose=pose, target_joints=final_joints,
        )

    def attach_payload(self, grip_width_mm: float) -> bool:
        """Tell the planner the gripper is carrying a part about ``grip_width_mm`` across.

        The caller knows how wide the jaws closed, which is a real measurement of the
        part at the grasp line. It knows nothing about how far the part extends beyond
        it, so the length and the lateral allowance come from
        ``safety.planning_world.payload``, where an operator declares the worst case
        their cell handles.

        It returns ``False`` where nothing was attached, which includes the ordinary
        case of a cell that uses no planner. A cell that cannot model its payload
        carries on and says so.
        """
        cfg = getattr(getattr(self.config.safety, "planning_world", None), "payload", None)
        if cfg is None or self._payload_config_declined() is not None:
            return False
        from src.robot.safety.planning.self_envelope import carried_part_box, hand_spheres

        length = float(cfg.length_mm)
        # The self filter covers the part from here on, whatever the planner answers: a part the
        # planner could not attach is still in the gripper and still in front of the cameras.
        self._attached_payload = (length, float(cfg.lateral_margin_mm))
        self._payload_in_planner = False
        # The planner carries the same part where the hand holds it: along the approach, from the fingertips.
        kinematics = self._preflight.self_kinematics(self)
        hand = self._preflight.planner_hand(self)
        spheres = hand_spheres(hand, kinematics[0]) if kinematics is not None and chosen(hand) else None
        if spheres is None or not chosen(hand):
            self.logger.error(
                "payload NOT attached: this cell's guard places no hand model on a known arm, so there is no "
                "fingertip to hang the part from; the planner is routing as if the gripper were empty"
            )
            return False
        dims_mm, centre_mm = carried_part_box(
            hand, spheres, grip_width_mm=grip_width_mm, length_mm=length,
            lateral_margin_mm=float(cfg.lateral_margin_mm),
        )
        try:
            joints = self.get_joint_positions().tolist()
        except RobotConnectionError:
            return False
        self._payload_in_planner = bool(self._curobo_ur_planner().attach_payload(joints, dims_mm, centre_mm))
        return self._payload_in_planner

    def _payload_config_declined(self) -> str | None:
        """Why this arm's configuration models no carried part, or ``None``; reads no hand and starts no planner."""
        cfg = getattr(getattr(self.config.safety, "planning_world", None), "payload", None)
        if cfg is None or not bool(cfg.enabled):
            return ("safety.planning_world.payload.enabled is false, so neither the planner nor the self filter "
                    "models a part in the gripper")
        if self._motion_planner != "curobo":
            return (f"robot.ur.motion_planner is {self._motion_planner!r}, and only the cuRobo planner carries a "
                    "part in the gripper")
        if cfg.length_mm is None:
            return ("safety.planning_world.payload.length_mm is undeclared, so neither the planner nor the self "
                    "filter models a part in the gripper: declare how far the longest part hangs past the fingertips")
        return None

    def payload_declined_reason(self) -> str | None:
        """Why this arm models no carried part, naming the key or the hand, or ``None`` where it can.

        The configuration first, then the hand the guard places: the same checks
        :meth:`attach_payload` makes, in the same order. It starts no planner and opens no
        connection.
        """
        declined = self._payload_config_declined()
        if declined is not None:
            return declined
        from src.robot.safety.planning.self_envelope import hand_spheres

        kinematics = self._preflight.self_kinematics(self)
        hand = self._preflight.planner_hand(self)
        if kinematics is None or not chosen(hand) or hand_spheres(hand, kinematics[0]) is None:
            return ("this cell's guard places no hand model on a known arm, so there is no fingertip to hang a part "
                    "from")
        return None

    def payload_model(self) -> PayloadModel:
        """What models the part the gripper carries now: what the last attach left in force."""
        if self._attached_payload is None:
            return PayloadModel.NONE
        return PayloadModel.PLANNER_AND_FILTER if self._payload_in_planner else PayloadModel.FILTER_ONLY

    def line_motion(self) -> LineReading:
        """What a ``move(pose, linear=True)`` on this arm keeps of the line, read before moving.

        On ik the controller draws the line with ``moveL`` and only the endpoint is gated. On
        cuRobo every sample is solved and judged by the exact mesh guard and the planner, and
        one ``moveL`` runs, unless the path cannot be judged, which the preflight says in the
        sentence its gate would refuse with.
        """
        if self._motion_planner != "curobo":
            return LineReading(LineMotion.CONTROLLER_LINE, (
                f"robot.ur.motion_planner is {self._motion_planner!r}: the controller draws the line and only its "
                "end is judged"))
        if self._preflight is None:
            return LineReading(LineMotion.NOT_KEPT, "no safety preflight is wired, so the line runs unjudged")
        refusal = self._preflight.path_judge_refusal(self)
        if refusal is not None:
            return LineReading(LineMotion.NOT_KEPT, refusal)
        return LineReading(LineMotion.CHECKED, (
            "every sample of the line is solved and judged by the exact mesh guard and the planner before one moveL "
            "runs"))

    def detach_payload(self) -> bool:
        """Forget the carried part. Safe to call when nothing was ever attached.

        The self filter forgets it here. The planner answers for its own model
        (:meth:`CuroboUrPlanner.detach_payload`): ``True`` without asking its sidecar where no attach was
        handed to it, as after a grasp that modelled nothing, and the sidecar's answer wherever one was.
        """
        self._attached_payload = None
        self._payload_in_planner = False
        if self._curobo_ur is None:
            return True
        return self._curobo_ur.detach_payload()

    def _curobo_ur_planner(self) -> CuroboUrPlanner:
        """Lazily build the real-UR cuRobo execution glue bound to this arm's RTDE connection."""
        if self._curobo_ur is None:
            from src.robot.safety.planning.reservation import PlannerReservation
            from src.robot.safety.planning.world import (
                build_planner_cuboids,
                build_planner_meshes,
            )

            from .curobo_motion import CuroboUrPlanner

            # The planner is built for this robot. A default factory taking no arguments
            # falls through to WILLY_CUROBO_ROBOT or ur5e.yml, so a real UR3e would be
            # planned against UR5e link lengths, on hardware, with nothing saying so. The
            # real-UR cuRobo path is fail-closed and never degrades to blind IK, so a
            # missing willy_{model}.yml stops the cell instead, which is the correct failure.
            #
            # The cell own geometry goes with it, or the planner routes through it.
            # Without this the planner knows its robot model and a generic table and
            # nothing about the bench, the bin walls or any fixture, so it plans straight
            # through them and the only thing that objects is the one-shot guard on the
            # final configuration, which cannot see a path. It is empty and disabled by
            # default, `safety.planning_world.enabled` turns it on, and the block refuses
            # to enable without a declared bench, because registering a world replaces
            # the planner own.
            world_cfg = getattr(self.config.safety, "planning_world", None)
            reservation = PlannerReservation.from_config(robot_cfg=self.config)
            self._curobo_ur = CuroboUrPlanner(
                self._conn,
                client_factory=self._curobo_client_factory or self._default_curobo_client_factory(),
                vel=self.config.motion_limits.max_velocity,
                acc=self.config.motion_limits.max_acceleration,
                world_cuboids=build_planner_cuboids(
                    world_cfg, self.config.safety.self_collision.fixtures,
                    max_cuboids=reservation.cuboid_slots,
                ),
                world_meshes=build_planner_meshes(world_cfg),
                reservation=reservation,
                require_registration=bool(
                    getattr(world_cfg, "require_registration", True)
                ),
                live_world=self._live_world,
                self_envelope=self._self_envelope,
                descriptor_check=self._descriptor_check(),
                # After a sent moveJ fails the arm may have moved part of the way, and the controller's own
                # state is what says whether a stop ended it.
                controller_state=self._controller_state_text,
                # The path guard lives on this arm rather than on the planner, and the two
                # have to be looking at the same cell: a planner routing around a tote the
                # guard cannot see gives a path that avoids it and a gate that would have
                # passed one straight through.
                on_perceived_obstacles=self._preflight.set_perceived_obstacles,
            )
            # The attach spheres the reservation counts, the same number the evidence lookup names.
            if reservation.sphere_slots:
                self._curobo_ur.enable_payload(int(reservation.sphere_slots))
        return self._curobo_ur

    @property
    def live_planner_world(self) -> "LivePlannerWorld | None":
        """What this arm asks before it plans, or `None` where nothing was handed in.

        Read by whatever holds the scene, which is the pick loop: it is the only thing that
        knows which object an attempt is reaching for, and that object has to be left out of
        the world or the planner refuses to approach it.
        """
        return self._live_world

    def set_wrist_bodies(self, bodies: "Sequence[Any]") -> None:
        """Hand this arm the wrist cameras it carries (``body_link.WristBody``), before its planner or
        guard is built.

        The planner loads them on top of the config its combination evidence names, the exact mesh guard
        and the self filter hold them, and the planner start proves their cover. Refused once the planner
        was built or the guard judged a path, because a camera handed in then would be one they say they
        hold and do not. On a ``willy`` cell each body's recorded flange to TCP has to be the declared
        frame; a ``polyscope`` cell compares it at connect. An empty sequence clears them.
        """
        if self._curobo_ur is not None:
            raise RuntimeError(
                "this arm's planner was already built, so a wrist camera handed in now would not be in it: hand the "
                "cell's wrist bodies to the arm before its first planned move or planner start"
            )
        listed = tuple(bodies)
        if self.config.gripper.tool_frame.source == "willy":
            for body in listed:
                stale = body.record_refusal("willy", self._declared_tool_matrix())
                if stale is not None:
                    raise ValueError(stale)
        self._preflight.set_wrist_bodies(listed)

    def set_live_planner_world(self, world: "LivePlannerWorld | None") -> None:
        """Hand this arm the thing that answers what the cell looks like right now.

        Set before the first planned move. An arm that already built its planner passes it
        on, so a cell that wires this after connecting still gets it.
        """
        self._live_world = world
        if self._curobo_ur is not None:
            self._curobo_ur.set_live_world(world)

    @property
    def camera_world_required(self) -> bool:
        """Whether every motion of this arm needs a live camera world or a decline: true on cuRobo."""
        return self._motion_planner == "curobo"

    def camera_world_for_move(self, camera_world: Maybe[CameraWorldDecline] = UNSET) -> CameraWorldStamp:
        """The camera world a :meth:`move` with ``camera_world`` would carry, read as the arm is now.

        The stamp :meth:`move` starts from before a refresh vouches for it: UNPLANNED on ik,
        MISSING on cuRobo with no live world and no decline, DECLINED for a decline. It reads no
        connection and starts no planner.
        """
        return self._move_camera_world(camera_world)

    def live_camera_world_wired(self) -> bool:
        """Whether a live camera world is wired to this arm."""
        return self._live_world is not None

    def without_camera_world(self, reason: str) -> AbstractContextManager[CameraWorldDecline]:
        """Decline the camera world for every motion this arm commands inside the ``with`` block.

        Bound to this arm alone (:func:`~src.robot.core.camera_world.without_camera_world`). A
        decline reaches both typed motions, :meth:`move` and :meth:`move_to_joints`, on the cuRobo
        planner, and on the ik planner they say UNPLANNED whatever was declined. With a live camera
        world wired a declined typed motion is refused before its path is judged, because every path
        on that arm is judged against the world.
        """
        return _without_camera_world(self, reason)

    def _self_envelope(self) -> "SelfEnvelope | None":
        """This arm's own body now, for taking the robot out of the view.

        A fixed camera watching a cell sees the robot, and a robot registered as an obstacle
        cannot move at all. The links are capsules fitted to the committed bundle, the hand is its
        sphere map and an attached part a capsule from the fingertips, placed with the guard's
        model and yaw. `None` where no model or no hand resolves or the connection cannot answer,
        and the world source then refuses to build a perceived world rather than registering the
        arm as geometry.
        """
        from src.robot.safety.planning.self_envelope import self_envelope

        try:
            joints = np.asarray(self._conn.get_joint_positions(), dtype=np.float64)
        except Exception:  # noqa: BLE001 (a cell that cannot say where it is has no self to filter)
            return None
        return self_envelope(self._preflight, self, joints, payload=self._attached_payload)

    def _default_curobo_client_factory(self) -> Callable[[], CuroboPlanClient]:
        """A cuRobo client bound to this cell robot model and to its guard planner margin.

        It mirrors the sim driver: the configured model wins, and the clearance the
        SelfCollisionGuard will demand is handed to the planner, so it stops returning
        configurations the guard would reject.
        """
        from src.robot.drivers.sim.robot_models import curobo_arm_descriptor
        from src.robot.safety.planning._curobo_body_links import BodyLinkError
        from src.robot.safety.planning.body_link import HandLink, coupling_bodies
        from src.robot.safety.planning.margin import planner_margin_refusal
        from src.robot.safety.planning.reservation import PlannerReservation
        from src.robot.safety.planning.robot.retract_table import (
            RetractMissing,
            read_retract,
        )

        # No implied margin. A UR cell that plans with cuRobo says what clearance its planner keeps,
        # or it does not plan: an undeclared margin read as 0 lets the sidecar propose paths at
        # 9.44 mm against a 10 mm guard. Read here and raised in build(), where a cell that never
        # plans never pays for it.
        margin_refusal = planner_margin_refusal(self.config)

        hand = self._preflight.planner_hand(self)
        # One descriptor per arm, with the hand this cell names added as a body link when the sidecar
        # starts. A cell that names no hand has no hand to add: the refusal is raised when the planner
        # first starts, which a move reports as CONTROLLER_REJECTED.
        robot_yml = curobo_arm_descriptor(self.config.ur.model) if chosen(hand) else None
        margin_mm = float(getattr(self.config.safety.self_collision, "planner_margin_mm", 0.0) or 0.0)
        # What the sidecar allocates, from this cell's own declaration. Without it the sidecar starts
        # with 16 boxes, no mesh and no grid whatever the cell declared, so a tote or a live scene meets
        # a planner with nowhere to put it.
        reservation = PlannerReservation.from_config(robot_cfg=self.config)

        def build() -> CuroboPlanClient:
            if margin_refusal is not None:
                raise CuroboUnavailableError(f"{margin_refusal[0]} {margin_refusal[1]}")
            if robot_yml is None or not chosen(hand):
                raise CuroboUnavailableError(
                    "robot.gripper.model is unset, so this cell has no hand to add to the planner's robot, and a "
                    "planner is never started for a hand nobody named"
                )
            try:
                link = HandLink.from_hand(hand)
            except BodyLinkError as exc:
                raise CuroboUnavailableError(str(exc)) from exc
            # The pose this pair was judged at. The descriptor is per arm and carries a fallback, while
            # the pose a cell plans from belongs to the arm and the hand on it, because a pose that
            # clears the exact meshes with one hand can be a self collision in the planner's sphere
            # model with another: on a ur3 the anchor works with the 2F-85 and refuses with the Hand-E.
            try:
                default_q = read_retract(str(self.config.ur.model), hand.model, float(hand.coupling_mm),
                                        margin_mm,
                                        placement=f"{link.placement.approach}{link.placement.closing}")
            except RetractMissing as exc:
                raise CuroboUnavailableError(str(exc)) from exc
            client = CuroboPlanClient(
                robot_config=robot_yml, self_collision_margin_mm=margin_mm,
                body_links=[link.to_dict(), *coupling_bodies(hand)],
                # The wrist cameras this arm was handed, read when the planner is built.
                wrist_body_links=[body.link() for body in self._preflight.wrist_bodies(self)],  # type: ignore[attr-defined]
                default_q=default_q,
            )
            client.reserve_world(reservation)
            return client

        return build

    def _descriptor_check(self) -> "Callable[[SidecarIdentity], str | None]":
        """What the planner asks of the sidecar it started: this arm's descriptor, no hand in it, this cell's hand added,
        and a committed evidence file that measured exactly that combination.

        The descriptor check comes first because it says which robot the sidecar loaded, and the evidence
        check then asks whether anybody measured that robot. A sidecar that reports no hashes is refused
        rather than trusted, because an unreported hash is not a hash that matched.
        """
        from src.robot.safety.planning._curobo_body_links import BodyLinkError
        from src.robot.safety.planning.body_link import HandLink, wrist_body_refusal
        from src.robot.safety.planning.evidence import planner_evidence_refusal
        from src.robot.safety.planning.hand import descriptor_refusal
        from src.robot.safety.planning.margin import declared_planner_margin
        from src.robot.safety.planning.reservation import PlannerReservation

        hand = self._preflight.planner_hand(self)
        model = str(self.config.ur.model)
        if not chosen(hand):
            return lambda identity: (
                "robot.gripper.model is unset, so nothing the planner loaded can be checked against the hand this cell "
                "carries"
            )
        try:
            link = HandLink.from_hand(hand)
        except BodyLinkError as exc:
            unplaced = str(exc)
            return lambda identity: unplaced
        sc = self.config.safety.self_collision
        declared = declared_planner_margin(sc)
        attach = PlannerReservation.from_config(robot_cfg=self.config).sphere_slots

        def check(identity: "SidecarIdentity") -> "str | None":
            said = descriptor_refusal(identity, link, arm=model)
            if said is not None:
                return said
            # The wrist cameras the sidecar loaded are exactly this arm's, and their cover is proven
            # complete: the evidence below is keyed without them.
            said = wrist_body_refusal(identity, self._preflight.wrist_bodies(self))  # type: ignore[arg-type]
            if said is not None:
                return said
            if not chosen(declared):
                return ("safety.self_collision.planner_margin_mm is undeclared, so no evidence can be looked up: "
                        "the margin is part of what a file measured")
            return planner_evidence_refusal(
                identity, hand, link, arm=model, planner_margin_mm=float(declared),
                guard_margin_mm=float(sc.min_distance_mm), attach_spheres=int(attach), mesh_dir=sc.mesh_dir,
            )

        return check

    async def amove_to(
        self,
        pose: Pose | URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Awaitable variant of :meth:`move_to`, gated and commanded the same way.

        On a cuRobo cell it runs the sync twin in a worker thread rather than mirroring its
        body. The plan and the sampled line are both synchronous and both talk to the
        sidecar and to the controller; a second async body beside them is how this verb
        drifted from its twin twice already.
        """
        if self._motion_planner == "curobo":
            return await asyncio.to_thread(
                self.move_to, pose, linear=linear, vel=vel, acc=acc, register=register,
            )
        rejected = self._gate_pose(self._pose_from_any(pose), MotionCommand.MOVE_TO)
        if rejected is not None:
            self.logger.error("amove_to REFUSED by %s: %s", rejected.status, rejected.message)
            return False
        return await self._motion.amove_to(
            self._to_controller_urpose(pose),
            linear=linear, vel=vel, acc=acc, register=register,
            workspace_pose=self._coerce_urpose(pose),
        )

    async def amove_home(self) -> bool:
        """The awaitable variant of :meth:`move_home`, gated the same way.

        Routing straight to ``MotionController.amove_home`` instead would log that it is
        proceeding anyway where the home pose lies outside the workspace box, and
        command the move regardless with no joint-limit, self-collision or payload check
        at all, which would leave this path strictly weaker than the method it mirrors.
        Both verbs run :meth:`move_to_home`, which runs synchronously.

        It runs the whole sync body in a worker thread rather than mirroring it: two bodies
        that have to stay the same is how this verb drifted apart from its twin in the
        first place.
        """
        return await asyncio.to_thread(self.move_home)

    def fk(self, joints: JointPositions) -> Pose:
        """Forward kinematics: :class:`JointPositions` to TCP :class:`Pose`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("fk() requires an open connection.")
        if joints.dof != self.capabilities.dof:
            raise RobotKinematicsError(
                f"fk() expected {self.capabilities.dof} DoF, got {joints.dof}."
            )
        try:
            tcp_m = self._conn.fk(joints.tolist())
        except (RuntimeError, OSError) as exc:
            raise RobotKinematicsError(f"FK failed: {exc}") from exc
        urpose = URPose.from_ur_list(tcp_m, label="fk")
        return self._pose_from_controller(urpose, label="fk")

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        """Inverse kinematics: TCP :class:`Pose` in :attr:`Frame.BASE` to joints."""
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"ik() requires pose in Frame.BASE; got {pose.frame!r}."
            )
        if not self._conn.is_connected:
            raise RobotConnectionError("ik() requires an open connection.")
        urpose = self._pose_to_controller(pose)
        seed_list = seed.tolist() if seed is not None else None
        try:
            joints = self._conn.ik(urpose.to_ur_list(), q_near=seed_list)
        except (RuntimeError, OSError) as exc:
            raise RobotKinematicsError(f"IK failed: {exc}") from exc
        if not joints or len(joints) != self.capabilities.dof:
            raise RobotKinematicsError(
                f"IK returned no valid solution for pose '{pose.label or '<unlabeled>'}'."
            )
        return JointPositions(joints)

    @property
    def capabilities(self) -> RobotCapabilities:
        """Static feature flags advertised by this driver."""
        return self._capabilities

    def get_tcp_pose(self) -> Pose:
        """Current TCP pose, tagged :attr:`Frame.BASE`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("get_tcp_pose() requires an open connection.")
        urpose = self._motion.get_current_pose()
        return self._pose_from_controller(urpose, label="current")

    def get_joint_positions(self) -> JointPositions:
        """Current joint configuration as a typed vector."""
        if not self._conn.is_connected:
            raise RobotConnectionError(
                "get_joint_positions() requires an open connection."
            )
        return JointPositions(self._conn.get_joint_positions())

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        camera_world: Maybe[CameraWorldDecline] = UNSET,
    ) -> MotionResult:
        """The typed joint move: judge it (:meth:`_judge_joint_move`), then drive exactly what was judged.

        On cuRobo the target goes onto the full turns nearest the arm inside the cable window first, the same
        pose; the straight joint line runs where both authorities pass it, and where it collides cuRobo plans
        around it. The result names the configuration that ran.

        The result says what stood behind the motion as :meth:`move` says it: UNPLANNED on the ik
        planner, and on cuRobo DECLINED for a decline, MISSING while no live camera world is wired,
        and once one is, PLANNED when the refresh this motion made vouched for the cell and UNSTATED
        when none did: the line and any plan around it are judged against that refreshed world. A
        declined joint move on an arm whose live camera world is wired is refused with ``UNSUPPORTED``
        before its path is judged, and so is a MISSING joint move, with neither a world nor a decline.
        """
        stamp = self._move_camera_world(camera_world)
        refused = self._refused_for_camera_world(stamp, MotionCommand.MOVE_JOINTS, target_joints=joints)
        if refused is not None:
            return refused
        before = self._planner_refresh()
        unstamped = self._move_to_joints_unstamped(
            joints, velocity=velocity, acceleration=acceleration,
        )
        after = self._planner_refresh()
        vouched = after.camera_world() if after is not None and after is not before else None
        return stamp_result(unstamped, self._move_camera_world(camera_world, planned=vouched))

    def _move_to_joints_unstamped(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """The body of :meth:`move_to_joints`. It stamps nothing; the public verb does."""
        target, route, refused = self._judge_joint_move(joints, command=MotionCommand.MOVE_JOINTS)
        if refused is not None:
            return refused
        return self._drive_judged_joints(
            target, route, command=MotionCommand.MOVE_JOINTS, done="move_to_joints",
            velocity=velocity, acceleration=acceleration,
        )

    def _drive_judged_joints(
        self,
        joints: JointPositions,
        route: "_Route | None",
        *,
        command: MotionCommand,
        done: str,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """Run the joint move that was just judged, and type what the controller answered.

        ``route`` is what :meth:`_judge_joint_move` passed: ``None`` or a straight line is one ``moveJ`` to
        ``joints``, the target that was judged, and a cuRobo plan is one ``moveJ`` per waypoint of the list that was
        judged. The ungated primitives, because the move was just judged; calling the public ``move_joint`` here
        would run the same guards a second time for nothing. A drive that fails is a result and never a raise:
        :meth:`_drive_joints` carries the typed refusal on its raise, whose message says whether ``moveJ`` had been
        sent, and a raise that carries none still reads CONTROLLER_REJECTED rather than a status nobody could
        classify. ``done`` is the message of a move that ran.
        """
        if route is not None:
            self._log_the_route(done, route)
        if route is not None and route.planned:
            vel, acc = self._motion._clamp(velocity, acceleration)
            result = self._curobo_ur_planner().execute(
                route.waypoints, command=command, target_joints=joints, vel=vel, acc=acc,
            )
            return dataclasses.replace(result, message=done) if result.ok else result
        try:
            self._drive_joints(joints, velocity=velocity, acceleration=acceleration, command=command)
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command,
                target_joints=joints, message=str(exc), exception=exc,
            )
        except RobotMotionRejected as exc:
            if isinstance(exc.result, MotionResult):
                return exc.result
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_joints=joints,
                message=f"the drive refused after the move was judged, and moveJ may have been sent: {exc}",
                exception=exc,
            )
        return MotionResult.executed(command, target_joints=joints, message=done)

    def _judge_joint_move(
        self, joints: JointPositions, *, command: MotionCommand
    ) -> "tuple[JointPositions, _Route | None, MotionResult | None]":
        """Judge a joint move before anything is commanded, as ``(target, route, refusal)``.

        ``refusal`` is ``None`` where the move may run, and then ``target`` is the configuration it ends at and
        ``route`` the judged way there, for :meth:`_drive_judged_joints`. On an ik cell ``target`` is ``joints``,
        ``route`` is ``None`` and the judge is the destination gate and nothing else: no planner produced a path,
        and ``moveJ`` interpolating between two configurations is a line nothing in this repository models.

        On a cuRobo cell it is the whole path from where the arm is standing, judged twice over, because the two
        authorities see different cells. The local guards carry the declared fixtures and the exact link meshes;
        the planner carries the camera world and whatever is attached to the tool. Either one alone would miss what
        the other holds. In order:

        1. A joint outside the window the planner and the joint-limit guard admit (the cable window about home where
           the cell keeps one) goes onto its full turn nearest the arm inside it (:meth:`_twin_near_the_arm`). A full
           turn of a joint is the same pose, so this changes which number is commanded and nothing about where the
           arm ends; where it changes one, the log says so.
        2. The destination guards on that target (``gate_joint_target``): a target no turn of which the window
           admits, or one past a full turn, which reads as degrees, is refused here by the joint-limit guard before
           any line is sampled or any plan asked for.
        3. The straight joint line to it, judged by both authorities, the planner at
           ``safety.planned_motion.line_clearance_mm`` from its world. Passing, it runs as one ``moveJ``.
        4. A line that collides (``SELF_COLLISION_REJECTED``, the status of the robot, the fixtures and the
           planner's world alike) is planned around by cuRobo in joint space, to the same target. The plan is taken
           where no joint swings more than ``safety.planned_motion.max_detour_deg`` beyond its span, else the move is
           refused ``JOINT_LIMIT_REJECTED`` naming the joint, and its legs meet both authorities as a Cartesian
           plan's do. Any other refusal of the line stands.

        The connection is read first, because a path starts at the current configuration and a disconnected arm has
        none to give. A read that fails is CONNECTION_ERROR too: the transport raises ``RuntimeError`` or ``OSError``
        rather than a robot error, and it would otherwise leave the verb as a raw exception. With a live camera world,
        the world is refreshed next, before either authority judges: the local guard learns the perceived obstacles
        from that refresh, and a refresh inside the planner's check would run after the guard, which would then judge
        against the motion before.
        """
        if self._preflight is None:
            return joints, None, None
        if self._motion_planner != "curobo":
            return joints, None, self._preflight.gate_joint_target(joints, arm=self)

        if not self._conn.is_connected:
            return joints, None, MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=joints,
                message="a judged joint move starts at the current configuration, which needs an "
                        "open connection.",
            )
        try:
            here = [float(v) for v in self._conn.get_joint_positions()]
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            return joints, None, MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=joints,
                message=(f"a judged joint move starts at the current configuration, and the controller's "
                         f"joint positions could not be read ({type(exc).__name__}: {exc}); nothing was sent"),
                exception=exc,
            )
        target, turned = self._twin_near_the_arm(joints, here)
        refused = self._refresh_before_the_path_gate(
            near_point_mm=self._flange_mm(target), command=command, target_joints=target,
            goal_tcp_mm=self._tcp_at_joints_mm(target),
        )
        if refused is not None:
            return target, None, refused
        refused = self._preflight.gate_joint_target(target, arm=self)
        if refused is not None:
            if any(not abs(float(v)) <= self._TWIN_MAX_RAD for v in target.tolist()):
                refused = dataclasses.replace(refused, message=(
                    f"{refused.message} The target holds a value past a full turn, which no UR joint reaches in "
                    "radians and reads as degrees; joint targets are radians."))
            return target, None, dataclasses.replace(refused, command=command)
        planner = self._curobo_ur_planner()
        line = [here, target.tolist()]
        refused = self._judge_legs(planner, line, command=command, target_joints=target,
                                   clearance_mm=self._line_clearance_mm())
        if refused is None:
            return target, self._route(line, planned=False, how="direct line" + turned), None
        if refused.status is not MotionStatus.SELF_COLLISION_REJECTED:
            return target, None, refused
        try:
            trajectory, why, start = self._planned(planner, target.tolist(), name="the target")
            if start is not None:
                return target, None, self._start_refusal(start, command, target_joints=target)
            if trajectory is None:
                return target, None, dataclasses.replace(
                    refused, message=f"{refused.message}; and no plan goes around it: {why}")
            _, detour = self._detour(trajectory)
            if detour:
                return target, None, MotionResult.failed(
                    MotionStatus.JOINT_LIMIT_REJECTED, command, target_joints=target,
                    message=(f"the straight line to this target is refused ({refused.message}), and cuRobo's plan "
                             f"around it {detour}; nothing was sent"),
                )
            waypoints, planned_refusal = self._judged_waypoints(
                planner, trajectory, command=command, target_joints=target,
            )
        except CuroboUnavailableError as exc:
            return target, None, MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_joints=target,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        if planned_refusal is not None:
            return target, None, planned_refusal
        how = f"cuRobo plan around the straight line, which was refused ({refused.status.value}: {refused.message})"
        return target, self._route(waypoints, planned=True, dense=len(trajectory), how=how + turned), None

    #: A joint value past this is past a full turn either way, which no UR joint reaches in radians: a target written in
    #: degrees by mistake. It is never turned onto a twin, so the joint-limit guard refuses it rather than the wrap
    #: making a plausible configuration out of it.
    _TWIN_MAX_RAD = 2.0 * math.pi + 1e-9

    def _twin_near_the_arm(self, joints: JointPositions, here: "Sequence[float]") -> "tuple[JointPositions, str]":
        """``joints`` with every joint outside the goal window on its full turn nearest ``here`` inside it, and what changed.

        The window is :meth:`_goal_joint_window`: the planner's, narrowed to the joint-limit guard's, which is the
        half turn either side of home where ``safety.joint_limits.within_half_turn_of_home`` is set. A full turn of a
        UR joint is the same pose, so a station stored as -200 degrees on a wrist runs as +160 inside a window of
        +-175 about home, rather than being refused for a cable it would not wind.

        What is left as given: a joint already inside the window, because that number is the turn the caller chose
        and the one the pendant will show (inside the cable window it is the only turn there is); a joint no turn of
        which lies inside the window, and a joint past a full turn either way (:data:`_TWIN_MAX_RAD`), a target written
        in degrees, both for the joint-limit guard to refuse; and a vector of the wrong length, for the drive to refuse.
        The second value is ``""`` where nothing changed, and otherwise a clause for the route's log line, which is
        also logged here.
        """
        values = [float(v) for v in joints.tolist()]
        lower, upper = self._goal_joint_window()
        if not len(values) == len(here) == len(lower):
            return joints, ""
        turned: dict[int, float] = {}
        for axis, (value, near, low, high) in enumerate(zip(values, here, lower, upper)):
            if low <= value <= high or not abs(value) <= self._TWIN_MAX_RAD:
                continue
            twin = nearest_turn(value, near=float(near), lower=low, upper=high)
            if twin is not None:
                turned[axis] = twin
        if not turned:
            return joints, ""
        from .curobo_motion import UR_ARM_JOINT_NAMES

        names = UR_ARM_JOINT_NAMES if len(values) == len(UR_ARM_JOINT_NAMES) else tuple(
            f"joint {i + 1}" for i in range(len(values)))
        said = ", ".join(f"{names[axis]} {math.degrees(values[axis]):.1f} -> {math.degrees(twin):.1f} deg"
                         for axis, twin in turned.items())
        self.logger.info("joint target turned onto the full turn inside the goal window nearest the arm, the same "
                         "pose: %s", said)
        for axis, twin in turned.items():
            values[axis] = twin
        return JointPositions(values), f"; twin wrap {said}"

    def _refresh_before_the_path_gate(
        self,
        *,
        near_point_mm: "list[float] | None",
        command: MotionCommand,
        target_joints: JointPositions | None = None,
        target_pose: Pose | None = None,
        goal_tcp_mm: "np.ndarray | None" = None,
    ) -> "MotionResult | None":
        """Refresh the live camera world before a path is judged; ``None`` means judging may start.

        ``goal_tcp_mm`` is the TCP at the path's goal, where the space between the jaws is left out
        of the world; ``None`` records why no region was placed.

        The local path guard learns the perceived obstacles from the refresh. Run inside the
        planner's check, after that guard, it would leave the guard judging this path against the
        obstacles of the motion before. It runs here instead, once, and the planner is asked with
        ``refresh=False``. Without a live world it does nothing, which is the unchanged path.

        A world that could not be refreshed refuses with CONTROLLER_REJECTED, the status of every
        planner refusal on this driver. A camera that stayed silent, blind or stale raises
        ``CameraWorldUnavailable`` out of here and out of the verb.
        """
        if self._live_world is None:
            return None
        try:
            self._curobo_ur_planner().refresh_world(near_point_mm=near_point_mm, **self._goal_keep_out(goal_tcp_mm))
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command,
                target_joints=target_joints, target_pose=target_pose,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        return None

    #: How far a plan's last configuration may put the flange from the flange goal it was planned
    #: to: the tolerance the Isaac driver holds an executed cuRobo plan to (``drivers/sim/arm.py``
    #: ``_MOVE_POS_TOL_MM``, ``_MOVE_ORI_TOL_DEG``). A base half a turn out misses by about a
    #: metre and 180 degrees, far outside either.
    _PLAN_END_TOL_MM = 5.0
    _PLAN_END_TOL_DEG = 6.0

    def _plan_end_refusal(self, goal: Pose, last_joints: "Sequence[float]", pose: Pose) -> "MotionResult | None":
        """A refusal when the plan's last configuration does not put the flange on ``goal``, or ``None`` when it does.

        Read on the DH chain, which is the controller's own frame, so a planner that answers in
        another base is caught here rather than on the robot. ``goal`` is the flange goal the
        planner was handed and ``pose`` the TCP pose the caller asked for, which the refusal names.
        """
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        model = str(self.config.ur.model)
        frames = ur_link_transforms_mm(model, np.asarray([float(v) for v in last_joints], dtype=np.float64))
        if frames is None:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=(f"robot.ur.model {model!r} has no DH chain here, so where this plan ends cannot be read, and "
                         "a plan whose end nobody read does not move"),
            )
        end = np.asarray(frames[-1], dtype=np.float64)
        target = np.asarray(goal.to_matrix(), dtype=np.float64)
        error_mm = float(np.linalg.norm(end[:3, 3] - target[:3, 3]))
        cosine = (float(np.trace(end[:3, :3].T @ target[:3, :3])) - 1.0) / 2.0
        error_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if error_mm <= self._PLAN_END_TOL_MM and error_deg <= self._PLAN_END_TOL_DEG:
            return None
        return MotionResult.failed(
            MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
            message=(f"the plan ends {error_mm:.1f} mm and {error_deg:.1f} deg from its goal (at most "
                     f"{self._PLAN_END_TOL_MM:g} mm and {self._PLAN_END_TOL_DEG:g} deg), read on the controller's own "
                     "kinematics, so nothing moved: a planner answering in another base than the controller's ends "
                     "here"),
        )

    def _goal_keep_out(self, goal_tcp_mm: "np.ndarray | None") -> "dict[str, Any]":
        """The goal region's keyword for a refresh, or nothing on an arm that holds no live world.

        A region is computed only where a world is refreshed, so an arm with no world keeps the
        unchanged path. An undeclared tool frame places no TCP, and the refresh records that no
        region was placed and why.
        """
        if self._live_world is None:
            return {}
        from src.robot.core.keep_out import GoalKeepOut
        from src.robot.safety.planning.self_envelope import goal_keep_out

        if goal_tcp_mm is None:
            reason = ("robot.gripper.tool_frame is undeclared, so the TCP at this goal is not known and no goal region "
                      "is placed") if self._declared_tool_matrix() is None else (
                "the TCP at this goal is not known, so no goal region is placed")
            return {"goal_keep_out": GoalKeepOut(region=None, reason=reason)}
        return {"goal_keep_out": goal_keep_out(self._preflight, self, goal_tcp_mm)}

    def _tcp_of_pose(self, pose: Pose) -> "np.ndarray | None":
        """A TCP goal as a 4x4, or ``None`` when the tool frame is undeclared and the pose is the flange's."""
        if self._declared_tool_matrix() is None:
            return None
        return np.asarray(pose.to_matrix(), dtype=np.float64)

    def _tcp_at_joints_mm(self, joints: JointPositions) -> "np.ndarray | None":
        """The TCP at ``joints``: the flange frame the chain ends in times the declared tool frame, or ``None``."""
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        tool = self._declared_tool_matrix()
        frames = ur_link_transforms_mm(str(self.config.ur.model), np.asarray(joints.tolist(), dtype=np.float64))
        if tool is None or frames is None:
            return None
        return np.asarray(frames[-1], dtype=np.float64) @ tool

    def _flange_mm(self, joints: JointPositions) -> "list[float] | None":
        """Where the flange stands at ``joints``, in BASE millimetres; ``None`` for a model with no chain.

        The near point of a joint move's refresh: when the slot budget bites, the obstacles nearest
        the goal are the ones kept, and the goal of a joint move is where its flange ends up.
        """
        from src.robot.safety._ur_kinematics import ur_link_origins_mm

        origins = ur_link_origins_mm(
            str(self.config.ur.model), np.asarray(joints.tolist(), dtype=np.float64)
        )
        return None if origins is None else [float(v) for v in origins[-1]]

    #: How much the tool may turn along a ``willy`` mode line before it stops being a
    #: straight line for the thing that matters. The controller runs the flange straight;
    #: with the tool composed on this side, a turn swings the grasp centre off that line by
    #: up to the tool length. Half a degree: enough to absorb a pose that is nominally
    #: equal, far too little to hide a real reorientation.
    _WILLY_LINE_MAX_TURN_DEG = 0.5

    def _judge_linear_move(
        self, pose: Pose, *, command: MotionCommand
    ) -> "MotionResult | None":
        """Judge the straight line the controller will run to ``pose``; ``None`` means it may run.

        On an ik cell this is the endpoint gate and nothing else.

        On a cuRobo cell the line is sampled in the frame the controller interpolates,
        because that is the frame in which the motion is actually straight: the bare flange
        in ``willy`` mode, the declared tool in ``polyscope`` mode. Every sample is solved by
        the controller's own inverse kinematics, seeded with the previous solution and the
        first with the joints the arm stands at, so the configurations judged are the ones the
        controller will pass through rather than a second opinion from a different solver. Then
        the local guards judge all of them, and the planner judges the same list against its
        world.

        It is not free: one RTDE round trip per sample, and the samples are as dense as the
        collision margin asks for. A line is the one motion in this stack that nothing plans,
        so the alternative to paying for it is executing it unexamined.
        """
        if self._preflight is None:
            return None
        if self._motion_planner != "curobo":
            return self._gate_pose(pose, command)
        if not self._conn.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_pose=pose,
                message="a judged line starts at the current pose, which needs an open connection.",
            )

        step_mm = self._preflight.path_step_mm
        radii = self._preflight.joint_radii_mm(self)
        if step_mm is None or radii is None:
            # The same two refusals a judged joint path gives, in the same words. Asked
            # through the joint gate rather than restated here, so one place says them.
            return self._preflight.gate_planned_path(
                [[float(v) for v in self._conn.get_joint_positions()]], arm=self, command=command,
            )

        owns_tool = self.config.gripper.tool_frame.source == "willy"
        start_tcp = self.get_tcp_pose()
        if owns_tool:
            turn_deg = float(
                np.degrees(angle_between(start_tcp.quaternion_xyzw, pose.quaternion_xyzw))
            )
            if turn_deg > self._WILLY_LINE_MAX_TURN_DEG:
                return MotionResult.failed(
                    MotionStatus.UNSUPPORTED, command, target_pose=pose,
                    message=(
                        f"this line turns the tool by {turn_deg:.2f} degrees, and in willy mode the "
                        f"controller runs the flange straight while this driver composes the tool: "
                        f"the grasp centre would travel an arc, not the line that was asked for. "
                        f"Move without the turn, or run the cell in polyscope mode, where the "
                        f"controller holds the tool and moveL draws a straight tool line"
                    ),
                )
            start_line = self._pose_to_flange(start_tcp)
            goal_line = self._pose_to_flange(pose)
        else:
            start_line, goal_line = start_tcp, pose

        try:
            samples = line_samples(
                start_line, goal_line, reach_mm=max(radii), max_step_mm=step_mm,
            )
        except ValueError as exc:
            return MotionResult.failed(
                MotionStatus.UNSUPPORTED, command, target_pose=pose, message=str(exc),
            )
        return self._judge_line_samples(
            samples, pose=pose, owns_tool=owns_tool, radii=radii, command=command,
        )

    def _line_step_refusal(
        self,
        previous: "Sequence[float]",
        solved: "Sequence[float]",
        *,
        index: int,
        count: int,
        bound_mm: float,
        pose: Pose,
        command: MotionCommand,
    ) -> "MotionResult | None":
        """Why the step from ``previous`` to ``solved`` is not part of one continuous line, or ``None`` when it is.

        A branch change is a joint jumping, or a chord off the line. It is not the sum
        |dq_j| * r_j against the step bound: that sum adds two joints turning against each other
        as if both carried the arm the same way, and on URSim a continuous lift sums to 27 mm a
        step while no link origin moves more than 10 mm. A jump is one joint turning more than a
        continuous line does. A branch change with no jump, the two elbow branches near full
        stretch, reaches the same flange pose from two configurations 0.30 rad apart, and it
        shows where the judged fill and moveL part: the joint line between them puts the flange
        off the line the controller runs.
        """
        where = "the joints the arm stands at" if index == 0 else f"sample {index}"
        turned = [abs(b - a) for a, b in zip(previous, solved)]
        joint = max(range(len(turned)), key=turned.__getitem__)
        if turned[joint] > LINE_MAX_JOINT_STEP_RAD:
            return MotionResult.failed(
                MotionStatus.IK_QUALITY_REJECTED, command, target_pose=pose,
                message=(
                    f"the controller changed IK branch between {where} and sample {index + 1} of {count} of this "
                    f"line: joint {joint + 1} turns {turned[joint]:.2f} rad in one step of at most {bound_mm:.1f} mm, "
                    f"where a continuous line turns no joint more than {LINE_MAX_JOINT_STEP_RAD} rad. The arm would "
                    f"take a route none of these samples describes"
                ),
            )
        model = str(self.config.ur.model)
        middle = [(a + b) / 2.0 for a, b in zip(previous, solved)]
        frames = [ur_link_transforms_mm(model, np.asarray(q, dtype=np.float64)) for q in (previous, solved, middle)]
        if any(chain is None for chain in frames):
            return MotionResult.failed(
                MotionStatus.UNSUPPORTED, command, target_pose=pose,
                message=(f"robot.ur.model {model!r} has no DH chain here, so whether this line's samples are one "
                         "branch cannot be read, and a line nobody read is not run"),
            )
        start, end, mid = (np.asarray(chain[-1], dtype=np.float64)[:3, 3] for chain in frames)  # type: ignore[index]
        off_mm = float(np.linalg.norm(mid - (start + end) / 2.0))
        if off_mm > LINE_MAX_CHORD_OFF_MM:
            return MotionResult.failed(
                MotionStatus.IK_QUALITY_REJECTED, command, target_pose=pose,
                message=(
                    f"the controller changed IK branch between {where} and sample {index + 1} of {count} of this "
                    f"line: no joint jumps, but the joint line between the two solutions puts the flange "
                    f"{off_mm:.2f} mm off the line moveL runs, more than {LINE_MAX_CHORD_OFF_MM} mm, so what would "
                    f"be judged is not what the controller would drive"
                ),
            )
        return None

    def _judge_line_samples(
        self,
        samples: "LineSamples",
        *,
        pose: Pose,
        owns_tool: bool,
        radii: "tuple[float, ...]",
        command: MotionCommand,
    ) -> "MotionResult | None":
        """Solve every sample of a line and hand the configurations to both authorities."""
        assert self._preflight is not None  # noqa: S101 (the caller checked)
        configs: list[tuple[float, ...]] = []
        # The line starts at the joints the arm stands at. moveL starts there whatever an
        # unseeded solution of the start pose says, so sample 0 is solved seeded on them and
        # judged against them like every later sample: a start on another branch is refused
        # rather than judged as if the arm were in it.
        seed: JointPositions = JointPositions([float(v) for v in self._conn.get_joint_positions()])
        bound = samples.step_bound_mm if samples.step_bound_mm > 0.0 else float(self._preflight.path_step_mm or 0.0)
        for index, sample in enumerate(samples.poses):
            tcp = self._pose_from_flange(sample) if owns_tool else sample
            if not self._guard.is_inside_workspace(self._coerce_urpose(tcp)):
                x, y, z = (float(v) for v in tcp.position_mm)
                return MotionResult.failed(
                    MotionStatus.WORKSPACE_REJECTED, command, target_pose=pose,
                    message=(
                        f"sample {index + 1} of {len(samples.poses)} of this line puts the grasp "
                        f"centre at ({x:.1f}, {y:.1f}, {z:.1f}), outside workspace_limits"
                    ),
                )
            try:
                solved = self.ik(tcp, seed=seed)
            except (RobotKinematicsError, RobotConnectionError) as exc:
                return MotionResult.failed(
                    MotionStatus.IK_FAILED, command, target_pose=pose,
                    message=(
                        f"sample {index + 1} of {len(samples.poses)} of this line has no inverse "
                        f"kinematics solution: {exc}"
                    ),
                    exception=exc,
                )
            previous = seed.tolist()
            refused = self._line_step_refusal(
                previous, solved.tolist(), index=index, count=len(samples.poses), bound_mm=bound, pose=pose,
                command=command,
            )
            if refused is not None:
                return refused
            # The path gate is told a bound on how far any point moves between two configurations
            # it judges, and the sum |dq_j| * r_j is the one bound that holds without the geometry.
            # Where a continuous step is more than the sum can vouch for, the joint line between
            # its two solutions is judged too, at the line's own step; the chord check above is
            # what says that joint line is the motion moveL runs.
            if bound > 0.0:
                try:
                    filled = joint_path_samples(previous, solved.tolist(), reach_mm=radii, max_step_mm=bound)
                except ValueError as exc:
                    return MotionResult.failed(
                        MotionStatus.UNSUPPORTED, command, target_pose=pose,
                        message=f"sample {index + 1} of {len(samples.poses)} of this line cannot be judged: {exc}",
                    )
                configs.extend(filled.configs[1:-1])
            configs.append(tuple(solved.tolist()))
            if len(configs) > MAX_PATH_SAMPLES:
                return MotionResult.failed(
                    MotionStatus.UNSUPPORTED, command, target_pose=pose,
                    message=(
                        f"this line needs more than {MAX_PATH_SAMPLES} configurations to be judged at "
                        f"{bound:.2f} mm a step; move it in shorter lines"
                    ),
                )
            seed = solved

        judged = PathSamples(configs=tuple(configs), step_bound_mm=bound)
        refused = self._refresh_before_the_path_gate(
            near_point_mm=[float(v) for v in self._pose_to_flange(pose).position_mm],
            command=command, target_pose=pose, goal_tcp_mm=self._tcp_of_pose(pose),
        )
        if refused is not None:
            return refused
        refused = self._preflight.gate_joint_path(judged, arm=self, command=command)
        if refused is not None:
            return refused
        try:
            verdict = self._curobo_ur_planner().check_joint_path(judged.configs, refresh=False)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_pose=pose,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        if verdict.valid:
            return None
        refusal = getattr(verdict, "refusal", None)
        return MotionResult.failed(
            _planner_refusal_status(refusal), command, target_pose=pose,
            message=_with_refusal(f"the planner refused this line: {verdict.reason}", refusal),
        )

    def _drive_checked_line(
        self, pose: Pose, *, vel: float | None = None, acc: float | None = None
    ) -> MotionResult:
        """Judge the line, then run it as one ``moveL``. The typed half of :meth:`move_linear`.

        The command that reaches the controller is the same one an unchecked linear move
        sent, through the same `MotionController.move_to`, so its endpoint inverse kinematics
        and its singularity check still run below this. What is added is that the line it
        draws has been looked at.

        A refusal below the judge carries the cause ``MotionController`` classified, as
        :meth:`move` does, and a sentence that says whether ``moveL`` had been sent.
        """
        refused = self._judge_linear_move(pose, command=MotionCommand.MOVE_TO)
        if refused is not None:
            return refused
        urpose = self._pose_to_controller(pose)
        ok = self._motion.move_to(
            urpose, linear=True, vel=vel, acc=acc, register=False,
            workspace_pose=self._coerce_urpose(pose),
        )
        if ok:
            return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)
        status = self._motion.last_reject_status
        before_movel = {
            MotionStatus.CONNECTION_ERROR: "the arm is not connected",
            MotionStatus.WORKSPACE_REJECTED: "its end is outside the controller's workspace box",
            MotionStatus.IK_FAILED: "the controller found no inverse kinematics solution for its end",
            MotionStatus.IK_QUALITY_REJECTED: "its end is near a singularity by the controller's own check",
        }
        if not isinstance(status, MotionStatus):
            status = None
        if status in before_movel:
            message = f"the judged line was refused before moveL was sent: {before_movel[status]}"
        elif status is MotionStatus.CONTROLLER_REJECTED:
            message = ("the judged line was sent as moveL and the controller reported it failed or raised; "
                       f"the arm may have moved part of the way.{self._controller_state_text()}")
        elif status is not None:
            message = f"MotionController.move_to refused the judged line as {status.value}; ur_motion.log has why"
        else:
            message = "MotionController.move_to refused the judged line and gave no cause; ur_motion.log has it"
        return MotionResult.failed(
            status if status is not None else MotionStatus.CONTROLLER_REJECTED,
            MotionCommand.MOVE_TO, target_pose=pose, message=message,
        )

    def _pose_from_flange(self, flange: Pose) -> Pose:
        """Flange (tool0) to TCP, the inverse of :meth:`_pose_to_flange`."""
        t = self._declared_tool_matrix()
        if t is None:
            return flange
        m = np.asarray(flange.to_matrix(), dtype=np.float64) @ t
        return Pose.from_matrix(m, frame=Frame.BASE, label=flange.label)

    def move_joint(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """A gated joint-space move through the UR RTDE driver.

        The Protocol promises ``RobotMotionRejected`` where a safety pre-check denies
        the move, and this is a public Protocol-level surface that sim runners and
        operator code reach directly rather than only through :meth:`move_to_joints`. It
        judges exactly what that method judges, through the same helper, and raises what
        it returns: the destination on an ik cell, the whole path on a cuRobo one, and it
        drives what was judged. On cuRobo, a move with neither a camera world nor a decline in
        scope is refused before the judge.
        """
        refused = self._refused_for_camera_world(
            self._move_camera_world(UNSET), MotionCommand.MOVE_JOINTS, target_joints=joints,
        )
        if refused is not None:
            raise RobotMotionRejected(f"move_joint refused: {refused.message}", result=refused)
        target, route, rejected = self._judge_joint_move(joints, command=MotionCommand.MOVE_JOINTS)
        if rejected is not None:
            if rejected.status is MotionStatus.CONNECTION_ERROR:
                raise RobotConnectionError(rejected.message or "move_joint needs a connection.")
            raise RobotMotionRejected(
                f"move_joint refused: {rejected.message}", result=rejected,
            ) from rejected.exception
        if route is None or not route.planned:
            if route is not None:
                self._log_the_route("move_joint", route)
            self._drive_joints(target, velocity=velocity, acceleration=acceleration)
            return
        result = self._drive_judged_joints(target, route, command=MotionCommand.MOVE_JOINTS, done="move_joint",
                                           velocity=velocity, acceleration=acceleration)
        if not result.ok:
            raise RobotMotionRejected(f"move_joint failed: {result.message}", result=result)

    def _drive_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        command: MotionCommand = MotionCommand.MOVE_JOINTS,
    ) -> None:
        """Command the joints with no safety gate. Callers must have gated the destination first.

        Each refusal raises with its typed result, labelled ``command``, so a typed verb returns
        it instead of a status nobody could classify. The message says whether ``moveJ`` had
        been sent. Once it was, a raise or a ``False`` does not mean the arm stayed where it
        was: a controller that stops a motion midway, on a protective stop for instance, stops
        it wherever it had got to. So those two messages also carry the controller's robot and
        safety state, where it can be read.
        """
        if not self._conn.is_connected:
            raise RobotConnectionError("move_joint() requires an open connection, and moveJ was not sent.")
        if joints.dof != self.capabilities.dof:
            message = f"move_joint() expected {self.capabilities.dof} DoF, got {joints.dof}, and moveJ was not sent."
            raise RobotMotionRejected(message, result=MotionResult.failed(
                MotionStatus.INVALID_TARGET, command, target_joints=joints, message=message,
            ))
        vel, acc = self._motion._clamp(velocity, acceleration)
        try:
            ok = self._conn.moveJ(joints.tolist(), vel=vel, acc=acc)
        except (RuntimeError, OSError) as exc:
            message = (f"moveJ failed: {exc}. moveJ was sent and raised, so the arm may have moved part of the "
                       f"way.{self._controller_state_text()}")
            raise RobotMotionRejected(message, result=MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=joints, message=message, exception=exc,
            )) from exc
        if not ok:
            message = ("Driver reported moveJ() failure. moveJ was sent and the controller did not complete it, "
                       f"so the arm may have moved part of the way.{self._controller_state_text()}")
            raise RobotMotionRejected(message, result=MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_joints=joints, message=message,
            ))

    def _controller_state_text(self) -> str:
        """The controller's robot and safety state as a sentence for a refusal, or ``""`` where it cannot be read.

        Best effort, and only ever read after a command failed: a state that cannot be read adds
        nothing to the refusal, and must not replace it with a second fault.
        """
        try:
            status = self.get_robot_status()
            sentence = (f" The controller reports robot mode {RobotMode(status.robot_mode).value} and safety "
                        f"mode {SafetyMode(status.safety_mode).value}")
            stops = [name for name, on in (("a protective stop", status.protective_stopped),
                                           ("an emergency stop", status.emergency_stopped)) if on is True]
            text = status.message.strip() if isinstance(status.message, str) else ""
        except Exception:  # noqa: BLE001 (best effort after a failure: no state is not a new fault)
            return ""
        if stops:
            sentence += f", with {' and '.join(stops)} active"
        if text:
            sentence += f" ({text!r})"
        return sentence + "."

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Cartesian move to ``pose`` in :attr:`Frame.BASE`.

        On cuRobo, a move with neither a camera world nor a decline in scope is refused before the
        connection is read.
        """
        refused = self._refused_for_camera_world(
            self._move_camera_world(UNSET), MotionCommand.MOVE_TO, target_pose=pose,
        )
        if refused is not None:
            raise RobotMotionRejected(f"move_linear refused: {refused.message}", result=refused)
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"move_linear() requires pose in Frame.BASE; got {pose.frame!r}."
            )
        if not self._conn.is_connected:
            raise RobotConnectionError("move_linear() requires an open connection.")
        rejected = self._judge_linear_move(pose, command=MotionCommand.MOVE_TO)
        if rejected is not None:
            raise RobotMotionRejected(
                f"move_linear refused: {rejected.message}", result=rejected,
            ) from rejected.exception
        urpose = self._pose_to_controller(pose)
        ok = self._motion.move_to(
            urpose, linear=True, vel=velocity, acc=acceleration, register=False,
            workspace_pose=self._coerce_urpose(pose),
        )
        if not ok:
            raise RobotMotionRejected(
                f"Linear move to '{pose.label or '<unlabeled>'}' was rejected."
            )

    @property
    def connection(self) -> URConnection:
        """Underlying RTDE connection.

        Ungated: its move calls reach the controller below the SafetyPreflight.
        """
        return self._conn

    @property
    def guard(self) -> WorkspaceGuard:
        """Workspace guard instance."""
        return self._guard

    @property
    def motion(self) -> MotionController:
        """Motion controller instance.

        Ungated: its move calls check only the workspace box, IK and singularity, below
        the SafetyPreflight.
        """
        return self._motion

    # ------------------------------------------------------------------
    # The optional capabilities: SupportsDigitalIO, SupportsForceTorque,
    # SupportsRobotStatus and SupportsFreedrive. They are not on the vendor-neutral RobotArm
    # Protocol, a caller feature-checks with ``isinstance(arm, SupportsForceTorque)``, and
    # every method delegates to the RTDE interfaces in URConnection. A non-UR driver does not
    # implement them.
    # ------------------------------------------------------------------

    # --- SupportsDigitalIO ---
    def set_digital_output(
        self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD
    ) -> None:
        """Drive a controller digital output pin (bank ``port``) high/low."""
        self._conn.set_digital_out(int(pin), bool(value), str(port))

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read a controller digital input pin (bank ``port``)."""
        return self._conn.get_digital_in(int(pin), str(port))

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read back a controller digital output pin's commanded level (bank ``port``)."""
        return self._conn.get_digital_out(int(pin), str(port))

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        """Set a controller analog output: voltage (V) unless ``current=True`` (A)."""
        self._conn.set_analog_out(int(pin), float(value), bool(current))

    # --- SupportsForceTorque ---
    def get_tcp_wrench(self) -> Wrench:
        """The live generalized force and torque at the TCP, in N and Nm in the BASE frame.

        It is the hand-over signal.
        """
        f = self._conn.get_tcp_force()
        return Wrench(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]),
                     frame=Frame.BASE)

    def get_joint_torques(self) -> tuple[float, ...]:
        """Live torque at each joint (newton-metres), base-to-tool order."""
        return tuple(float(t) for t in self._conn.get_joint_torques())

    # --- SupportsRobotStatus + enriched errors ---
    def get_robot_status(self) -> RobotStatus:
        """A snapshot of the controller live robot mode, safety mode, stop flags and text.

        It is the vendor-neutral projection of the UR ``getRobotMode``,
        ``getSafetyMode``, ``isProtectiveStopped`` and ``isEmergencyStopped``, plus the
        dashboard ``safetystatus`` text: the enriched controller state a pipeline
        surfaces instead of a bare motion rejection.
        """
        return RobotStatus(
            robot_mode=_UR_ROBOT_MODE.get(self._conn.get_robot_mode(), RobotMode.UNKNOWN),
            safety_mode=_UR_SAFETY_MODE.get(self._conn.get_safety_mode(), SafetyMode.UNKNOWN),
            protective_stopped=self._conn.is_protective_stopped(),
            emergency_stopped=self._conn.is_emergency_stopped(),
            message=self._conn.dashboard_safety_status(),
        )

    def recover_from_protective_stop(self) -> bool:
        """Ask the controller (via the dashboard) to release an active protective stop."""
        ok = self._conn.unlock_protective_stop()
        if ok:
            self.logger.info("Requested protective-stop release from the UR dashboard.")
        else:
            self.logger.warning("Protective-stop release unavailable (no dashboard connection).")
        return ok

    # --- SupportsFreedrive ---
    def freedrive(self) -> FreedriveSession:
        """A hand-guiding session on the UR teach mode (``SupportsFreedrive``); nothing is freed yet.

        Refused while the arm is not connected, while the controller is not operational (a stop, a
        safety mode other than normal, brakes engaged) and while a session is open. The session frees
        the arm on ``free()`` inside its with-block, and from the with-block's start to its end every
        motion verb of this arm refuses; :mod:`.freedrive` has the order of the controller calls.
        """
        if not self._conn.is_connected:
            raise RobotConnectionError("freedrive() requires an open connection.")
        if self._freedrive is not None:
            raise RobotMotionRejected(f"{HAND_GUIDED_MESSAGE}; leave it before opening another")
        raise_unless_operational(self.get_robot_status(), "a freedrive session")
        return URFreedriveSession(
            self._conn, tcp_pose=self.get_tcp_pose, robot_status=self.get_robot_status,
            on_open=self._freedrive_opened, on_close=self._freedrive_closed,
        )

    def controller_payload(self) -> ControllerPayload | None:
        """The payload the controller compensates for, or ``None`` where RTDE does not report it.

        It decides whether a freed arm floats, sinks or rises, and it is the controller's own record,
        which is the one ``safety.payload`` pushed at connect only where that says ``enforce: true``.
        """
        read = self._conn.get_payload()
        if read is None:
            return None
        mass_kg, cog_mm = read
        return ControllerPayload(mass_kg=mass_kg, cog_mm=cog_mm)

    def _freedrive_opened(self, session: URFreedriveSession) -> None:
        """Record ``session`` as the open one; another open session is refused."""
        if self._freedrive is not None and self._freedrive is not session:
            raise RobotMotionRejected(f"{HAND_GUIDED_MESSAGE}; leave it before opening another")
        self._freedrive = session

    def _freedrive_closed(self, session: URFreedriveSession) -> None:
        """Forget ``session``, which gives motion back."""
        if self._freedrive is session:
            self._freedrive = None

    def _hand_guided_refusal(
        self, command: MotionCommand, *,
        target_pose: Pose | None = None, target_joints: JointPositions | None = None,
    ) -> MotionResult | None:
        """The refusal of every motion while a freedrive session is open, or ``None``.

        The one check every motion verb meets: :meth:`_refused_for_camera_world` asks it for every verb,
        and :meth:`_gate_pose` for the ik path of ``move_to`` and ``amove_to``, both before a planner or
        the controller is asked anything.
        """
        if self._freedrive is None:
            return None
        return MotionResult.failed(
            MotionStatus.CONTROLLER_REJECTED, command, target_pose=target_pose, target_joints=target_joints,
            message=(f"{HAND_GUIDED_MESSAGE}, so this arm does not drive itself and no command reached the "
                     f"controller; leave the session first, which holds the arm and gives motion back"),
        )

    def raise_if_stopped(self) -> None:
        """Raise an enriched :class:`RobotEmergencyStop` where the controller is stopped or faulted.

        It guards a critical operation: it reads :meth:`get_robot_status` and, where the
        arm is in a protective or emergency stop or its safety mode is a stop state,
        raises with the safety mode and the controller own text, so the operator sees
        why it stopped rather than only that it did.
        """
        status = self.get_robot_status()
        if status.is_stopped:
            detail = f": {status.message}" if status.message else ""
            raise RobotEmergencyStop(
                f"UR controller in {status.safety_mode.value} "
                f"(robot_mode={status.robot_mode.value}){detail}"
            )
