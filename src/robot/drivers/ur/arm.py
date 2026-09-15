"""Universal Robots :class:`RobotArm` implementation."""

from __future__ import annotations

import asyncio
import dataclasses
import re

import numpy as np

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from src.config.schema.robot import RobotConfig
from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, FrameMismatchError, Pose
from src.robot.constants import HOME_JOINTS_DEFAULT, UR_ARM_LOG_FILE, create_robot_logger
from src.robot.core import (
    DECLINE_ON_A_LIVE_WORLD_MESSAGE,
    NO_PLAN_FAIL_SAFE_MESSAGE,
    CameraWorldDecline,
    CameraWorldStamp,
    CameraWorldUse,
    DigitalIOPort,
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
    resolve_camera_world,
    stamp_result,
)
from src.robot.core.camera_world import without_camera_world as _without_camera_world
from src.geometry.quaternion import angle_between
from src.robot.safety import (
    LineSamples,
    PathSamples,
    SafetyPreflight,
    line_samples,
)
from src.robot.safety._ur_kinematics import ur_series_twin
from src.robot.safety.planning import CuroboPlanClient, CuroboUnavailableError
from src.robot.safety.planning.hand import planner_hand
from src.robot.safety.workspace import WorkspaceGuard

from .connection import URConnection
from .motion import MotionController
from .pose import URPose
from .pose_adapter import pose_to_urpose, urpose_to_pose

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.safety.planning.live_world import (
        LivePlannerWorld,
        WorldRefresh,
    )
    from src.robot.safety.planning.perceived import SelfEnvelope
    from collections.abc import Callable

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
        self._curobo_client_factory = curobo_client_factory
        self._curobo_ur: CuroboUrPlanner | None = None
        #: The cell as the cameras see it, asked immediately before every plan. Handed in by
        #: whatever composes the cell rather than built here: a driver may not reach up into
        #: perception, and this one only ever asks. `None` is the unchanged path, where the
        #: declared world is registered once and the planner keeps it for the life of the cell.
        self._live_world: "LivePlannerWorld | None" = None

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
            # ⚠ SAY WHAT WAS NOT CHECKED. A size class is not a model: a UR3 and a UR3e are both
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
        self.logger.info(
            "tool frame verified: source=%s, controller matches within %.2f mm / %.2f deg",
            tf.source, d_t, d_r,
        )

    #: How much better the twin has to explain the controller before this refuses, in mm. Small
    #: on purpose: the two hypotheses are not close. A correct cell reads about 0 mm for its own
    #: model and 8.5 mm or more for the twin (the measured ur3/ur3e minimum over 20000 poses),
    #: so anything above sensor noise separates them. This is NOT a tolerance on being right; it
    #: is the margin by which the wrong answer has to win before the cell is refused.
    _TWIN_MARGIN_MM = 1.0

    def _refuse_if_the_series_twin_fits_better(
        self, joints: list[float], controller_tcp, expected, own_d_t: float
    ) -> None:
        """Refuse when the OTHER series of this size explains the controller better than we do.

        ⛔ **THE CHECK THE TOLERANCE CANNOT DO.** ``_verify_controller_model`` compares size
        classes, because a controller reports ``UR3`` for a UR3e. ``_verify_tool_frame`` compares
        a distance against a tolerance, and MEASURED over 20000 random joint vectors on
        2026-09-09, a ur3/ur3e swap lands under the default 10 mm tolerance in 12.2 % of poses
        (minimum separation 8.50 mm). So the two arms this cell is most likely to confuse are
        the two arms neither check can separate.

        This one is RELATIVE and therefore immune to the tolerance: the flange -> TCP transform
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

    def disconnect(self) -> None:
        """Close the RTDE connection, and shut down the cuRobo planning server where one was started."""
        if self._curobo_ur is not None:
            self._curobo_ur.close()
            self._curobo_ur = None
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
        move.
        """
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
        """Move to the home joint configuration, gated. ``False`` means refused, with nothing moved.

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

        It returns ``bool`` so every existing caller keeps working, and the typed reason
        is logged.

        On a cuRobo cell the whole line to the home configuration is judged as well, by
        :meth:`_judge_joint_move`, and the ``moveJ`` is sent here rather than through
        ``MotionController.move_home``: that method sends an unclamped command at the
        controller's own defaults, and a move that was judged has to be the move that runs.
        """
        if not self._home_is_admissible("move_home"):
            return False
        joints = JointPositions(np.asarray(self._home_joints, dtype=np.float64))
        refused = self._judge_joint_move(joints, command=MotionCommand.MOVE_HOME)
        if refused is not None:
            self.logger.error("move_home REFUSED by %s: %s", refused.status, refused.message)
            return False
        try:
            self._drive_joints(joints)
        except (RobotConnectionError, RobotMotionRejected) as exc:
            self.logger.error("move_home REFUSED: %s", exc)
            return False
        return True

    def _home_is_admissible(self, verb: str) -> bool:
        """The whole home gate, shared by :meth:`move_home` and :meth:`amove_home`.

        It returns ``False`` when the move is refused. The destination guards run first,
        then the workspace box on the home pose. The box bounds the grasp centre, so the
        controller FK is converted out of its TCP register first: in ``willy`` mode that
        register is the bare flange, and boxing it would pass a home whose grasp centre
        lies a tool length outside. An FK that cannot be read refuses, because an
        unchecked home is what this gate stops.
        """
        joints = JointPositions(np.asarray(self._home_joints, dtype=np.float64))
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                self.logger.error("%s REFUSED by %s: %s", verb, rejected.status, rejected.message)
                return False
        if not self._conn.is_connected:
            return True
        try:
            reported = URPose.from_ur_list(self._conn.fk(list(self._home_joints)), label="home")
            home_pose = self._coerce_urpose(self._pose_from_controller(reported, label="home"))
            if not self._guard.is_inside_workspace(home_pose):
                self.logger.error(
                    "%s REFUSED: the home grasp centre (%.1f, %.1f, %.1f) is outside "
                    "workspace_limits. Set robot.home_joint_positions to a configuration inside "
                    "this cell's own box; the shipped default was authored for a UR5e.",
                    verb, home_pose.x, home_pose.y, home_pose.z,
                )
                return False
        except Exception as exc:  # noqa: BLE001 (FK unavailable must not silently wave the move through)
            self.logger.error("%s REFUSED: could not check the home pose (%s)", verb, exc)
            return False
        return True

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
        cannot set it aside for one motion.
        """
        stamp = self._move_camera_world(camera_world)
        if stamp.use is CameraWorldUse.DECLINED and self._live_world is not None:
            return MotionResult.failed(
                MotionStatus.UNSUPPORTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=DECLINE_ON_A_LIVE_WORLD_MESSAGE, camera_world=stamp,
            )
        before = self._planner_refresh()
        unstamped = self._move_unstamped(pose, linear=linear, vel=vel, acc=acc, register=register)
        after = self._planner_refresh()
        vouched = after.camera_world() if after is not None and after is not before else None
        return stamp_result(unstamped, self._move_camera_world(camera_world, planned=vouched))

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
        """Plan from tool0 to ``pose`` with cuRobo, gate the planned final configuration, execute.

        It is fail-closed. An unavailable cuRobo gives ``CONTROLLER_REJECTED``, no
        collision-free plan gives ``TIMEOUT``, and a static safety-guard rejection of
        the actual final configuration gives the matching status. There is no blind
        motion.
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
        try:
            traj_ur = planner.plan(goal)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        if not traj_ur:
            return MotionResult.failed(
                MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose,
                message=NO_PLAN_FAIL_SAFE_MESSAGE,
            )
        # The workspace box belongs here too. `gate_joint_target` deliberately skips it,
        # and that is right for a joint-only command: the guard reads `target_pose`, so
        # on a joint context it accepts for want of a Cartesian target and un-skipping it
        # there would do nothing. At this gate the commanded pose is in hand, so a pose
        # context carrying the cuRobo final joints gets all four static guards: the box
        # on the target, joint limits and self-collision on the configuration the planner
        # returned, and payload. That is what the sim path does.
        #
        # The continuity guards stay skipped for the reason they are skipped in the sim:
        # cuRobo owns trajectory continuity, and a large net joint delta such as a
        # shoulder wrap from +pi to -pi is not a blind interpolator teleporting.
        #
        # The limit is real: the box is applied to the commanded TCP pose, not to the FK
        # of the planned configuration and not to the middle of the path. The path gate
        # below covers the path with the joint-readable guards, and no guard bounds a
        # planned configuration against the Cartesian box.
        #
        # The endpoint comes first, deliberately. It is the cheap check and the one most
        # often refused, while the path gate below costs a few milliseconds per sample of a
        # path with as many samples as the plan is long. Nothing moves before both have
        # passed, so the order is only about what a refused move pays.
        rejected = self._gate_planned_config(pose, JointPositions(traj_ur[-1]))
        if rejected is not None:
            return rejected
        # And the whole path. The gate above judges where the move ends, and a plan that
        # grazes a fixture in the middle and lands clear is exactly what an endpoint check
        # cannot see. The legs between the planner's waypoints are sampled, because the
        # controller moveJ's from one to the next and the line between them is executed
        # too. Judged before the first waypoint moves: a refusal partway leaves the arm
        # standing on a path already judged unsafe.
        rejected = self._preflight.gate_planned_path(
            traj_ur, arm=self, command=MotionCommand.MOVE_TO,
        )
        if rejected is not None:
            return rejected
        return planner.execute(traj_ur, pose, vel=vel, acc=acc)

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
        if cfg is None or not bool(cfg.enabled) or self._motion_planner != "curobo":
            return False
        from src.robot.safety.planning.self_envelope import carried_part_box, hand_spheres

        length = float(cfg.length_mm)
        # The self filter covers the part from here on, whatever the planner answers: a part the
        # planner could not attach is still in the gripper and still in front of the cameras.
        self._attached_payload = (length, float(cfg.lateral_margin_mm))
        # The planner carries the same part where the hand holds it: along the approach, from the fingertips.
        kinematics = self._preflight.self_kinematics(self)
        hand = self._preflight.planner_hand(self)
        spheres = hand_spheres(hand, kinematics[0]) if kinematics is not None and chosen(hand) else None
        if spheres is None:
            self.logger.error(
                "payload NOT attached: this cell's guard places no hand model on a known arm, so there is no "
                "fingertip to hang the part from; the planner is routing as if the gripper were empty"
            )
            return False
        dims_mm, centre_mm = carried_part_box(
            spheres, grip_width_mm=grip_width_mm, length_mm=length, lateral_margin_mm=float(cfg.lateral_margin_mm),
        )
        try:
            joints = self.get_joint_positions().tolist()
        except RobotConnectionError:
            return False
        return self._curobo_ur_planner().attach_payload(joints, dims_mm, centre_mm)

    def detach_payload(self) -> bool:
        """Forget the carried part. Safe to call when nothing was ever attached."""
        self._attached_payload = None
        if self._curobo_ur is None:
            return True
        return self._curobo_ur.detach_payload()

    def _curobo_ur_planner(self) -> CuroboUrPlanner:
        """Lazily build the real-UR cuRobo execution glue bound to this arm's RTDE connection."""
        if self._curobo_ur is None:
            from src.robot.safety.planning.reservation import PlannerReservation
            from src.robot.safety.planning.world import build_planner_cuboids, build_planner_meshes

            from .curobo_motion import CuroboUrPlanner

            # The planner is built for this robot. A default factory taking no arguments
            # falls through to WILLY_CUROBO_ROBOT or ur5e.yml, so a real UR3e would be
            # planned against UR5e link lengths, on hardware, with nothing saying so. The
            # real-UR cuRobo path is fail-closed and never degrades to blind IK, so a
            # missing {model}_{hand}.yml stops the cell instead, which is the correct failure.
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
                # The path guard lives on this arm rather than on the planner, and the two
                # have to be looking at the same cell: a planner routing around a tote the
                # guard cannot see gives a path that avoids it and a gate that would have
                # passed one straight through.
                on_perceived_obstacles=self._preflight.set_perceived_obstacles,
            )
            payload_cfg = getattr(world_cfg, "payload", None)
            if payload_cfg is not None and bool(getattr(payload_cfg, "enabled", False)):
                self._curobo_ur.enable_payload(int(payload_cfg.sphere_slots))
        return self._curobo_ur

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

        Set before the first planned move. An arm that already built its planner passes it
        on, so a cell that wires this after connecting still gets it.
        """
        self._live_world = world
        if self._curobo_ur is not None:
            self._curobo_ur.set_live_world(world)

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
        except Exception:  # noqa: BLE001 - a cell that cannot say where it is has no self to filter
            return None
        return self_envelope(self._preflight, self, joints, payload=self._attached_payload)

    def _default_curobo_client_factory(self) -> Callable[[], CuroboPlanClient]:
        """A cuRobo client bound to this cell robot model and to its guard planner margin.

        It mirrors the sim driver: the configured model wins, and the clearance the
        SelfCollisionGuard will demand is handed to the planner, so it stops returning
        configurations the guard would reject.
        """
        from src.robot.drivers.sim.robot_models import curobo_robot_yml
        from src.robot.safety.planning.reservation import PlannerReservation

        hand = self._preflight.planner_hand(self)
        # Named by the arm and the hand. A cell that names no hand has no descriptor to name: the
        # refusal is raised when the planner first starts, which a move reports as CONTROLLER_REJECTED.
        robot_yml = curobo_robot_yml(self.config.ur.model, hand.model) if chosen(hand) else None
        margin_mm = float(getattr(self.config.safety.self_collision, "planner_margin_mm", 0.0) or 0.0)
        # What the sidecar allocates, from this cell's own declaration. Without it the sidecar starts
        # with 16 boxes, no mesh and no grid whatever the cell declared, so a tote or a live scene meets
        # a planner with nowhere to put it.
        reservation = PlannerReservation.from_config(robot_cfg=self.config)

        def build() -> CuroboPlanClient:
            if robot_yml is None:
                raise CuroboUnavailableError(
                    "robot.gripper.model is unset, so this cell has no cuRobo descriptor to plan with: descriptors are "
                    "named by the arm and the hand, and a planner is never started for a hand nobody named"
                )
            client = CuroboPlanClient(robot_config=robot_yml, self_collision_margin_mm=margin_mm)
            client.reserve_world(reservation)
            return client

        return build

    def _descriptor_check(self) -> "Callable[[object], str | None]":
        """What the planner asks of the descriptor the sidecar loaded: this arm, the hand this cell names, its plate."""
        from src.robot.safety.planning.hand import descriptor_refusal

        hand = self._preflight.planner_hand(self)
        model = str(self.config.ur.model)
        if not chosen(hand):
            return lambda provenance: (
                "robot.gripper.model is unset, so no descriptor can be checked against the hand this cell carries"
            )
        return lambda provenance: descriptor_refusal(provenance, hand, arm=model)

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
        Both verbs share :meth:`_home_is_admissible`, which runs synchronously.

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
        """The typed joint move: gate the destination through the preflight, then drive.

        The result says what stood behind the motion as :meth:`move` says it: UNPLANNED on the ik
        planner, and on cuRobo DECLINED for a decline, MISSING while no live camera world is wired,
        and once one is, PLANNED when the refresh this motion made vouched for the cell and UNSTATED
        when none did. Nothing plans a joint move on this driver; on cuRobo its path is checked
        against that refreshed world, which is what PLANNED says here. A declined joint move on an
        arm whose live camera world is wired is refused with ``UNSUPPORTED`` before its path is
        judged.
        """
        stamp = self._move_camera_world(camera_world)
        if stamp.use is CameraWorldUse.DECLINED and self._live_world is not None:
            return MotionResult.failed(
                MotionStatus.UNSUPPORTED, MotionCommand.MOVE_JOINTS, target_joints=joints,
                message=DECLINE_ON_A_LIVE_WORLD_MESSAGE, camera_world=stamp,
            )
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
        refused = self._judge_joint_move(joints, command=MotionCommand.MOVE_JOINTS)
        if refused is not None:
            return refused
        # The ungated primitive, because the move was just judged. Calling the public
        # `move_joint` here would run the same guards a second time for nothing.
        try:
            self._drive_joints(joints, velocity=velocity, acceleration=acceleration)
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_JOINTS,
                target_joints=joints, message=str(exc), exception=exc,
            )
        return MotionResult.executed(
            MotionCommand.MOVE_JOINTS, target_joints=joints, message="move_to_joints",
        )

    def _judge_joint_move(
        self, joints: JointPositions, *, command: MotionCommand
    ) -> "MotionResult | None":
        """Judge a joint move before anything is commanded; ``None`` means it may run.

        On an ik cell this is the destination gate and nothing else: no planner produced a
        path, and `moveJ` interpolating between two configurations is a line nothing in this
        repository models.

        On a cuRobo cell it is the whole line from where the arm is standing to where it was
        told to go, judged twice over, because the two authorities see different cells. The
        local guards carry the declared fixtures and the exact link meshes; the planner
        carries the camera world and whatever is attached to the tool. Either one alone
        would miss what the other holds.

        The connection is read first, because a path starts at the current configuration and
        a disconnected arm has none to give. That is a change of order: gating the
        destination first only finds out afterwards that it cannot move.

        With a live camera world, the world is refreshed next, before either authority judges. The
        local guard learns the perceived obstacles from that refresh, and a refresh inside the
        planner's check would run after the guard, which would then judge against the motion before.
        """
        if self._preflight is None:
            return None
        if self._motion_planner != "curobo":
            return self._preflight.gate_joint_target(joints, arm=self)

        if not self._conn.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, command, target_joints=joints,
                message="a judged joint move starts at the current configuration, which needs an "
                        "open connection.",
            )
        here = [float(v) for v in self._conn.get_joint_positions()]
        refused = self._refresh_before_the_path_gate(
            near_point_mm=self._flange_mm(joints), command=command, target_joints=joints,
        )
        if refused is not None:
            return refused
        refused = self._preflight.gate_planned_path(
            [here, joints.tolist()], arm=self, command=command,
        )
        if refused is not None:
            return refused
        return self._planner_judges_joint_path(here, joints, command=command)

    def _planner_judges_joint_path(
        self, here: list[float], goal: JointPositions, *, command: MotionCommand
    ) -> "MotionResult | None":
        """The planner's verdict on the same line, against the world it holds; ``None`` accepts."""
        samples = self._preflight.path_step_mm if self._preflight is not None else None
        if samples is None:  # no self-collision guard: gate_planned_path already refused
            return None
        from src.robot.safety.path_samples import joint_path_samples

        reach = self._preflight.joint_radii_mm(self)
        if reach is None:
            return None  # gate_planned_path refused this already; here it would be a second voice
        path = joint_path_samples(here, goal.tolist(), reach_mm=reach, max_step_mm=samples)
        try:
            verdict = self._curobo_ur_planner().check_joint_path(path.configs, refresh=False)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command, target_joints=goal,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        if verdict.valid:
            return None
        return MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, command, target_joints=goal,
            message=f"the planner refused this joint path: {verdict.reason}",
        )

    def _refresh_before_the_path_gate(
        self,
        *,
        near_point_mm: "list[float] | None",
        command: MotionCommand,
        target_joints: JointPositions | None = None,
        target_pose: Pose | None = None,
    ) -> "MotionResult | None":
        """Refresh the live camera world before a path is judged; ``None`` means judging may start.

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
            self._curobo_ur_planner().refresh_world(near_point_mm=near_point_mm)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, command,
                target_joints=target_joints, target_pose=target_pose,
                message=f"cuRobo planner unavailable: {exc}", exception=exc,
            )
        return None

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
        the controller's own inverse kinematics, seeded with the previous solution, so the
        configurations judged are the ones the controller will pass through rather than a
        second opinion from a different solver. Then the local guards judge all of them, and
        the planner judges the same list against its world.

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
        seed: JointPositions | None = None
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
            if seed is not None:
                swept = sum(
                    abs(b - a) * radius
                    for a, b, radius in zip(seed.tolist(), solved.tolist(), radii)
                )
                if swept > samples.step_bound_mm + 1e-6:
                    return MotionResult.failed(
                        MotionStatus.IK_QUALITY_REJECTED, command, target_pose=pose,
                        message=(
                            f"the controller changed IK branch between samples {index} and "
                            f"{index + 1} of this line: {swept:.1f} mm of arm travel for a step "
                            f"bounded at {samples.step_bound_mm:.1f} mm. The arm would take a route "
                            f"none of these samples describes"
                        ),
                    )
            configs.append(tuple(solved.tolist()))
            seed = solved

        judged = PathSamples(configs=tuple(configs), step_bound_mm=samples.step_bound_mm)
        refused = self._refresh_before_the_path_gate(
            near_point_mm=[float(v) for v in self._pose_to_flange(pose).position_mm],
            command=command, target_pose=pose,
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
        return MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, command, target_pose=pose,
            message=f"the planner refused this line: {verdict.reason}",
        )

    def _drive_checked_line(
        self, pose: Pose, *, vel: float | None = None, acc: float | None = None
    ) -> MotionResult:
        """Judge the line, then run it as one ``moveL``. The typed half of :meth:`move_linear`.

        The command that reaches the controller is the same one an unchecked linear move
        sent, through the same `MotionController.move_to`, so its endpoint inverse kinematics
        and its singularity check still run below this. What is added is that the line it
        draws has been looked at.
        """
        refused = self._judge_linear_move(pose, command=MotionCommand.MOVE_TO)
        if refused is not None:
            return refused
        urpose = self._pose_to_controller(pose)
        ok = self._motion.move_to(
            urpose, linear=True, vel=vel, acc=acc, register=False,
            workspace_pose=self._coerce_urpose(pose),
        )
        return MotionResult.from_bool(ok, MotionCommand.MOVE_TO, target_pose=pose)

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
        it returns: the destination on an ik cell, the whole line on a cuRobo one.
        """
        rejected = self._judge_joint_move(joints, command=MotionCommand.MOVE_JOINTS)
        if rejected is not None:
            if rejected.status is MotionStatus.CONNECTION_ERROR:
                raise RobotConnectionError(rejected.message or "move_joint needs a connection.")
            raise RobotMotionRejected(
                f"move_joint refused: {rejected.message}", result=rejected,
            ) from rejected.exception
        self._drive_joints(joints, velocity=velocity, acceleration=acceleration)

    def _drive_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Command the joints with no safety gate. Callers must have gated the destination first."""
        if not self._conn.is_connected:
            raise RobotConnectionError("move_joint() requires an open connection.")
        if joints.dof != self.capabilities.dof:
            raise RobotMotionRejected(
                f"move_joint() expected {self.capabilities.dof} DoF, got {joints.dof}."
            )
        vel, acc = self._motion._clamp(velocity, acceleration)
        try:
            ok = self._conn.moveJ(joints.tolist(), vel=vel, acc=acc)
        except (RuntimeError, OSError) as exc:
            raise RobotMotionRejected(f"moveJ failed: {exc}") from exc
        if not ok:
            raise RobotMotionRejected("Driver reported moveJ() failure.")

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Cartesian move to ``pose`` in :attr:`Frame.BASE`."""
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
    # The optional capabilities: SupportsDigitalIO, SupportsForceTorque and
    # SupportsRobotStatus. They are not on the vendor-neutral RobotArm Protocol, a caller
    # feature-checks with ``isinstance(arm, SupportsForceTorque)``, and every method
    # delegates to the RTDE interfaces in URConnection. A non-UR driver does not
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
