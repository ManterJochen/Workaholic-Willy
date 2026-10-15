"""Universal Robots :class:`RobotArm` implementation."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Sequence

import numpy as np

from src.config.arms import load_arm
from src.config.schema.robot import RobotConfig
from src.geometry import Frame, FrameMismatchError, Pose
from src.robot.constants import UR_ARM_LOG_FILE, create_robot_logger
from src.robot.core import (
    JointPositions,
    RobotCapabilities,
    RobotConnectionError,
)
from src.robot.safety import (
    SafetyPreflight,
)
from src.robot.safety._ur_kinematics import URDhChain, ur_link_transforms_mm, ur_series_twin
from src.robot.safety.planning import (
    CuroboPlanClient,
)
from src.robot.safety.planning.mounted_tool import MountedTool
from src.robot.safety.workspace import WorkspaceGuard
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.drivers.planned.capabilities import (
    PlannedControllerProvenance,
    PlannedDigitalIO,
    PlannedFaultAcknowledge,
    PlannedForceTorque,
    PlannedFreedrive,
    PlannedRobotStatus,
    PlannedWatching,
)

from .connection import URConnection
from .motion import MotionController
from .pose import URPose
from .pose_adapter import pose_to_urpose, urpose_to_pose
from .transport import URTransport

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Callable, Sequence
    from typing import Any



__all__ = ["URRobotArm"]


def _capabilities_of(config: RobotConfig) -> RobotCapabilities:
    """What this arm is, read from the description of the model the tree names and from the address it declares.

    The record is what downstream code reads to learn which robot it is holding (a hardcoded "ur5e" once made every
    model-keyed lookup resolve the wrong robot for a real UR3e), so it is built before anything that asks: the
    preflight, the planner and the lock all read it. ``robot.ur.model`` is admitted at load, so the description exists.
    """
    description = load_arm(config.ur.model)
    return RobotCapabilities(
        vendor="ur",
        model=config.ur.model,
        dof=description.dof,
        controller=config.ur.ip,
        owns_a_controller=True,
        has_native_fk=True,
        is_simulated=False,
    )


class URRobotArm(
    PlannedDigitalIO, PlannedForceTorque, PlannedRobotStatus, PlannedFaultAcknowledge,
    PlannedControllerProvenance, PlannedFreedrive, PlannedWatching, PlannedArm,
):
    """The UR-backed arm driver.

    It satisfies the vendor-neutral :class:`RobotArm` Protocol and keeps the UR
    convenience API the current pipelines use.
    """

    #: The UR transport, which the RTDE-reading UR side below reaches into (``_conn``, ``_motion``).
    _transport: URTransport

    def __init__(
        self,
        config: RobotConfig,
        home_joints: list[float] | None = None,
        *,
        curobo_client_factory: Callable[[], CuroboPlanClient] | None = None,
    ) -> None:
        # Built in the order it always was, then handed to PlannedArm by keyword (P0g slice 4), so a tree that is refused
        # (a model with no description, a hand the exact-mesh guard cannot name) is refused by the same step as before.
        logger = create_robot_logger("URRobotArm", UR_ARM_LOG_FILE)
        # Who this arm is comes first: the preflight, the planner and the lock read it (P0e E4).
        capabilities = _capabilities_of(config)
        description = load_arm(capabilities.model)

        # The halt latch lives on this connection, which the arm keeps for its life, so a disconnect and a connect of
        # the same arm keep it; brake_on_halt decides whether a halt also brakes the move in flight.
        # The address the tree declares, or none: there is no placeholder any more (2026-10-13). A driver with no
        # address builds, so a desk can read this cell and say what it lacks, and refuses at `connect`, the first thing
        # that needs a controller to exist.
        connection = URConnection(
            ip=config.ur.ip or "",
            vel=config.motion_limits.max_velocity,
            acc=config.motion_limits.max_acceleration,
            frequency=config.ur.rtde_frequency,
            brake_on_halt=config.ur.brake_on_halt,
            # How "the arm stands still" is read: the controller's is_steady() or the joint speeds (2026-10-09).
            steady_signal=config.safety.dwell.steady_signal,
        )
        guard = WorkspaceGuard(config.workspace_limits)
        # The controller below the planning half (P0e E7): RTDE and the UR motion controller behind the verbs of
        # ``robot.core.transport``. ``_conn`` and ``_motion`` read and write through it.
        transport = URTransport(connection, MotionController(
            connection,
            guard,
            max_velocity=config.motion_limits.max_velocity,
            max_acceleration=config.motion_limits.max_acceleration,
            # The line's singularity check on the controller's FK or on the arm's own chain (2026-10-09).
            singularity_fk=config.safety.ik_quality.singularity_fk,
            arm_model=capabilities.model,
        ))
        # The vendor-neutral safety preflight pipeline. It owns its own WorkspaceGuard,
        # shrunk by the configured ``limits.workspace_margin_mm``, so the typed
        # ``move()`` path enforces the tighter operator-defined margin while
        # ``MotionController`` keeps the full-box guard as a backstop for the bool path.
        # The tool on the flange, read once (P0i i5): the guard models its hand, and the arm reads its tool frame
        # and its payload from it. The hand resolves where it always did, so a hand this tree cannot resolve is
        # refused here.
        tool = MountedTool.from_config(config)
        preflight = SafetyPreflight.from_safety_config(
            config.safety, config.workspace_limits,
            # The hand the cell names and the arm the guard places it on. An exact mesh guard with no
            # hand named refuses here, at build, rather than checking whatever hand the arm bundle carries.
            hand=tool.hand, arm_model=capabilities.model,
        )
        super().__init__(
            config=config,
            logger=logger,
            capabilities=capabilities,
            joint_names=tuple(joint.name for joint in description.joints),
            controller_yaw_deg=description.base.controller_yaw_deg,
            transport=transport,
            guard=guard,
            preflight=preflight,
            # An explicit argument wins over the cell's own config, which wins over the home its description states. A
            # real cell that declares nothing still gets a home pose, and move_home boxes it before it moves.
            home_joints=(
                list(home_joints) if home_joints
                else list(config.home_joint_positions or [joint.home_rad for joint in description.joints])
            ),
            curobo_client_factory=curobo_client_factory,
            tool=tool,
        )
        if self._next_leg_mode == "in_settles_and_motion" and not bool(config.ur.brake_on_halt):
            self.logger.warning("robot.motion.judge_next_leg is 'in_settles_and_motion' and robot.ur.brake_on_halt is "
                                "false: a leg is judged during the line before it only where a halt brakes that line, "
                                "so this arm judges ahead in the jaws' strokes alone ('in_settles')")

    # ------------------------------------------------------------------
    # Tool frame (flange <-> TCP)
    # ------------------------------------------------------------------
    #
    # Every Pose crossing the RobotArm Protocol is the TCP, meaning the grasp centre.
    # What the controller calls the TCP depends on its tool register, and
    # ``gripper.tool_frame.source`` declares who owns that fact:
    #   "willy"     the controller runs a bare flange and this driver composes the
    #               transform, as the Isaac driver does. No vendor SDK call is involved.
    #   "controller" the controller holds it, and poses pass through untouched.
    # Either way connect() derives what the controller is really using and refuses a
    # mismatch.

    @property
    def _conn(self) -> URConnection:
        """The RTDE connection, held by the transport. Writing it replaces the transport's, so an injection lands."""
        return self._transport.connection

    @_conn.setter
    def _conn(self, value: URConnection) -> None:
        self._transport.connection = value

    @property
    def _motion(self) -> MotionController:
        """The UR motion controller of the ik route, held by the transport; writing it replaces the transport's."""
        return self._transport.motion

    @_motion.setter
    def _motion(self, value: MotionController) -> None:
        self._transport.motion = value

    def _pose_from_controller(self, urpose: URPose, *, label: str) -> Pose:
        """Controller frame -> TCP, for a UR register reading: the conversion the transport makes, then the tool."""
        return self._tcp_from_controller(urpose_to_pose(urpose, frame=Frame.BASE, label=label), label=label)

    def _pose_to_controller(self, pose: Pose) -> URPose:
        """TCP -> the UR register pose the controller is told, as the transport converts it."""
        return pose_to_urpose(self._controller_pose(pose))

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
        reported_raw = self._transport.controller_model()
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

    #: How near, millimetres and degrees, the controller's rows have to put the flange to where the controller says it
    #: stands, at the joints it stands at, for the guard to place the flange by them. The controller computes its poses on
    #: those rows, so they answer it to a micrometre; rows that are not the ones it computes on part from it by
    #: millimetres (URSim's calibration of the line IK test: 3.2 mm).
    _FLANGE_VOUCHED_MM = 0.1
    _FLANGE_VOUCHED_DEG = 0.01

    def _vouched_flange_chain(self, joints: "Sequence[float]", controller_tcp_mm: np.ndarray) -> "URDhChain | None":
        """The UR's answer to the hook :meth:`PlannedArm._vouched_flange_chain`: the controller's own rows
        (``URConnection.controller_kinematics``) where they put the flange where the
        controller says it stands, its pose at ``joints`` (``controller_tcp_mm``, read while the arm stood) less its
        active TCP (``URConnection.active_tcp_offset``), within :data:`_FLANGE_VOUCHED_MM` and
        :data:`_FLANGE_VOUCHED_DEG`: the chain :meth:`controller_flanges_mm` answers by until the next connect.

        ``None``, and the exact guard places the flange by the nominal table as before, where the rows or the active TCP
        cannot be read, or the rows put the flange anywhere else. Said once a connect, either way, with how far the
        controller's flange stands from the nominal table's.
        """
        from src.robot.safety._ur_ik import ur_pose_matrix_m

        from src.geometry.matrix import compare_tool_frames

        from .connection import ControllerKinematics

        model = str(self._capabilities.model)
        nominal = (f"the exact guard places the flange, and the hand, its fingers and a wrist camera on it, by the nominal "
                   f"{model} table")
        # Read off the transport's connection where it has one: a transport that hands over no connection (the tests'
        # fake) has no rows to read, and the flange stands on the nominal table, as below where the rows are missing.
        connection = getattr(self._transport, "connection", None)
        read_rows = getattr(connection, "controller_kinematics", None)
        read_tcp = getattr(connection, "active_tcp_offset", None)
        try:
            kinematics = read_rows() if callable(read_rows) else None
            tcp = read_tcp() if callable(read_tcp) else None
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            self.logger.warning("%s: the controller's kinematics could not be read (%s)", nominal, exc)
            return None
        if not isinstance(kinematics, ControllerKinematics) or not isinstance(tcp, list) or len(tcp) != 6:
            self.logger.warning("%s: the controller's rows or its active TCP could not be read", nominal)
            return None
        here = np.asarray([[float(v) for v in joints]], dtype=np.float64)
        try:
            chain = URDhChain(theta_rad=kinematics.theta_rad, a_m=kinematics.a_m, d_m=kinematics.d_m,
                              alpha_rad=kinematics.alpha_rad)
            offset = ur_pose_matrix_m(tcp)
            offset[:3, 3] *= 1000.0
            theirs = np.asarray(controller_tcp_mm, dtype=np.float64) @ np.linalg.inv(offset)
            ours = chain.flanges_mm(here)[0]
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.logger.warning("%s: the controller's rows read as no chain (%s)", nominal, exc)
            return None
        apart_mm, apart_deg = compare_tool_frames(ours, theirs)
        if apart_mm > self._FLANGE_VOUCHED_MM or apart_deg > self._FLANGE_VOUCHED_DEG:
            self.logger.warning("%s: the controller's rows put the flange %.3f mm and %.4f deg from where the controller "
                                "says it stands, so they are not the ones it computes on", nominal, apart_mm, apart_deg)
            return None
        table = ur_link_transforms_mm(model, here[0])
        if table is not None:
            off_mm, off_deg = compare_tool_frames(ours, np.asarray(table[-1], dtype=np.float64))
            self.logger.info("the exact guard places the flange, and the hand, its fingers and a wrist camera on it, "
                             "where the controller's own kinematics put it (read from port %d): at the joints the arm "
                             "stands at, %.2f mm and %.3f deg from the nominal %s table", kinematics.port, off_mm,
                             off_deg, model)
        return chain

    #: How much better the twin has to explain the controller before this refuses, in mm. Small
    #: on purpose: the two hypotheses are not close. A correct cell reads about 0 mm for its own
    #: model and 8.5 mm or more for the twin (the measured ur3/ur3e minimum over 20000 poses),
    #: so anything above sensor noise separates them. This is NOT a tolerance on being right; it
    #: is the margin by which the wrong answer has to win before the cell is refused.
    _TWIN_MARGIN_MM = 1.0

    def _refuse_if_another_model_fits_better(
        self, joints: list[float], controller_tcp: np.ndarray, expected: np.ndarray, own_d_t: float
    ) -> None:
        """Refuse where the other series of this size explains the controller better than this one does (the UR's
        answer to the hook :meth:`PlannedArm._refuse_if_another_model_fits_better`).

        The check the tolerance cannot do. ``_verify_controller_model`` compares size classes,
        because a controller reports ``UR3`` for a UR3e. ``PlannedArm._verify_tool_frame`` compares a
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
        from src.geometry.matrix import compare_tool_frames

        from ..planned.tool_frame import derive_active_tool_frame

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

    def _address_refusal(self) -> str | None:
        """A UR cell declares the controller it drives, or it has nothing to connect to (the UR's answer to the hook
        :meth:`PlannedArm._address_refusal`; the rest of connect is ``PlannedArm.connect``)."""
        if self.config.ur.ip:
            return None
        return (
            "this cell drives a UR controller and robot.ur.ip declares no address for it, so there is nothing to "
            "connect to. Declare the controller address in the cell's own tree (robot.ur.ip), which is also the "
            "name the cell lock holds it by. Until 2026-10-13 this key defaulted to 192.168.1.100, so a tree "
            "nobody had commissioned still named a controller."
        )

    def is_inside_workspace(self, pose: Pose | URPose) -> bool:
        """The vendor-neutral workspace pre-check the pipelines use.

        It accepts a :class:`Pose` in the BASE frame, or a :class:`URPose`.
        """
        return self._guard.is_inside_workspace(self._boxed(pose))

    @classmethod
    def _boxed(cls, pose: Pose | URPose) -> Pose:
        """``pose`` as the workspace box judges it: the commanded TCP as a base-frame :class:`Pose`.

        The URPose round trip is the one the guard made itself until P0e, kept so every verdict stays
        bit-identical; the guard speaks :class:`Pose` only since then.
        """
        urpose = cls._coerce_urpose(pose)
        return urpose_to_pose(urpose, frame=Frame.BASE, label=urpose.label or None)

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

    def _home_fk(self, joints: JointPositions) -> Pose:
        """The controller's FK of the home ``joints``, the pose the home gate boxes, read again on a fresh control
        program where the program had ended.

        An FK that cannot be read because the control program on the controller ended, while the controller is in no
        stop (a stop on the pendant, a protective stop a person cleared), is read again on a fresh program
        (:meth:`_fk_on_a_fresh_program`): the home is a person's command, and a fresh program moves nothing. On the
        cell (2026-10-12) every home after such a stop was refused until the cell was taken down and connected again.
        A controller still in a protective or emergency stop is named, with what clears it, and the home gate refuses.
        """
        try:
            return self._transport.controller_fk(joints.tolist(), label="home")
        except Exception as exc:  # noqa: BLE001 (read again on a fresh program, or refused by the home gate)
            return self._fk_on_a_fresh_program(joints, exc)

    def _fk_on_a_fresh_program(self, joints: JointPositions, failed: BaseException) -> Pose:
        """The controller's FK of ``joints`` read again where the control program had ended (:meth:`_home_fk`).

        Where the controller stands in a protective or emergency stop, or its program still runs, or its state cannot
        be read, nothing is restarted and an error says so; a fresh program is started only where the program ended
        with the controller in no stop (``URConnection.restart_control_script``, about 2 s), and the FK read once more.

        Args:
            joints (JointPositions): The home configuration.
            failed (BaseException): What the first read raised.

        Returns:
            Pose: The FK the controller answered on the fresh program, in its own frame (the transport's reading).

        Raises:
            RuntimeError: Nothing restarted, or the fresh program did not answer either; it names what was found.
        """
        conn = self._conn
        try:
            protective = conn.is_protective_stopped()
            emergency = conn.is_emergency_stopped()
            running = conn.is_program_running()
        except Exception as exc:  # noqa: BLE001 (the state could not be read: nothing is restarted)
            raise RuntimeError(f"{failed}; the controller's state could not be read ({exc})") from failed
        if not all(isinstance(value, bool) for value in (protective, emergency, running)):
            raise RuntimeError(str(failed)) from failed
        if protective or emergency:
            which = "a protective stop" if protective else "an emergency stop"
            raise RuntimeError(f"{failed}; the controller is in {which}: clear it at the pendant, then ask for home "
                               "again") from failed
        if running:
            raise RuntimeError(str(failed)) from failed
        self.logger.warning("move_home: the control program on the controller had ended (a stop on the pendant, or a "
                            "stop that was cleared); a fresh one is started for this home a person asked for, and "
                            "nothing moves before every gate has run")
        try:
            conn.restart_control_script()
            return self._transport.controller_fk(joints.tolist(), label="home")
        except Exception as exc:  # noqa: BLE001 (refused by the caller, with both causes)
            raise RuntimeError(f"{failed}; a fresh control program did not answer either ({type(exc).__name__}: "
                               f"{exc})") from exc

    def _controller_chain(self) -> "tuple[Any, np.ndarray, str] | None":
        """The controller's own kinematics, for a line solved here (``robot.safety.ik_quality.line_ik: local``): its DH
        rows (a :class:`~src.robot.safety._ur_kinematics.URDhChain`), the inverse of its active TCP, and the model
        whose nominal table the closed form starts from; ``None`` where the rows or the TCP cannot be read, and the
        controller solves the line as before.

        Both are the connection's reads, once per connection (:meth:`URConnection.controller_kinematics`,
        :meth:`URConnection.active_tcp_offset`), which log what they read or why they could not. Once per connection
        this logs how far the controller's rows put the flange from the nominal table at the joints the arm stands at:
        on a calibrated arm, the part of the gap between the nominal chain and the controller's TCP that its
        calibration explains.
        """
        from src.robot.safety._ur_ik import ur_pose_matrix_m
        from src.robot.safety._ur_kinematics import URDhChain

        from .connection import ControllerKinematics

        # A transport that hands over no connection (the tests' fake) has no rows: the controller solves the line.
        connection = getattr(self._transport, "connection", None)
        if connection is None or not connection.is_connected:
            return None
        read_rows = getattr(connection, "controller_kinematics", None)
        read_tcp = getattr(connection, "active_tcp_offset", None)
        try:
            kinematics = read_rows() if callable(read_rows) else None
            tcp = read_tcp() if callable(read_tcp) else None
        except (RobotConnectionError, RuntimeError, OSError) as exc:
            self.logger.warning("the controller's kinematics could not be read (%s): lines are solved by the controller",
                                exc)
            return None
        if not isinstance(kinematics, ControllerKinematics) or not isinstance(tcp, list) or len(tcp) != 6:
            return None
        model = self._capabilities.model
        try:
            rows = URDhChain(theta_rad=kinematics.theta_rad, a_m=kinematics.a_m, d_m=kinematics.d_m,
                             alpha_rad=kinematics.alpha_rad)
            flange_from_tcp = np.linalg.inv(ur_pose_matrix_m(tcp))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.logger.warning("the controller's kinematics read as no chain to solve on (%s): lines are solved by the "
                                "controller", exc)
            return None
        if getattr(self, "_chain_described", None) is not kinematics:
            self._chain_described = kinematics
            nominal = URDhChain.nominal(model)
            try:
                here = [float(v) for v in self._conn.get_joint_positions()]
                ours, table = rows.flange_m(here), nominal.flange_m(here) if nominal is not None else None
            except (RobotConnectionError, RuntimeError, OSError, ValueError):
                table = None
            if table is not None:
                cosine = (float(np.trace(ours[:3, :3].T @ table[:3, :3])) - 1.0) / 2.0
                self.logger.info(
                    "lines are solved here on the controller's own kinematics (robot.safety.ik_quality.line_ik: local) "
                    "and checked against the controller at two samples each; at the joints the arm stands at, its rows "
                    "put the flange %.2f mm and %.3f deg from the nominal %s table",
                    1000.0 * float(np.linalg.norm(ours[:3, 3] - table[:3, 3])),
                    math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), model,
                )
        return rows, flange_from_tcp, model

    # ------------------------------------------------------------------
    # The optional capabilities: the mixins this class lists, each over a verb of URTransport (P0g
    # slices 6 and 7: the I/O, the force readings, the status, the fault release, the provenance),
    # and hand guiding (port S4, over ControllerFreedrive). None is
    # on the vendor-neutral RobotArm Protocol; a caller feature-checks with
    # ``isinstance(arm, SupportsForceTorque)``, and a vendor whose transport offers none lists none.
    # ------------------------------------------------------------------

    # --- SupportsForceTorque ---
    # --- SupportsRobotStatus + enriched errors ---
    def _controller_goal_m(self, tcp: Pose) -> np.ndarray:
        """The pose the controller is told for ``tcp`` (:meth:`_pose_to_controller`), as a 4x4 in metres through the
        UR's axis-angle, the goal a line solved here reads (``line_ik: local``)."""
        from src.robot.safety._ur_ik import ur_pose_matrix_m

        return ur_pose_matrix_m(self._pose_to_controller(tcp).to_ur_list())

