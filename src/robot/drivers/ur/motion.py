"""MotionController, the safe high-level UR motion execution.

It wraps ``URConnection`` with workspace-guard checks, so every Cartesian move is
validated before it reaches the controller.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from src.geometry import Frame, Pose
from src.robot.constants import HOME_JOINTS_DEFAULT, UR_MOTION_LOG_FILE, create_robot_logger
from src.robot.core import JointPositions, MotionStatus
from src.robot.drivers.ur import URConnection
from src.robot.drivers.ur.connection import HALT_ENDS
from src.robot.drivers.ur.pose import URPose
from src.robot.drivers.ur.pose_adapter import urpose_to_pose
from src.robot.safety.singularity import (
    SingularityThresholds,
    analyze_joint_singularity,
)
from src.robot.safety.workspace import WorkspaceGuard

if TYPE_CHECKING:
    import numpy as np

    from src.robot.core import RobotArm

__all__ = ["MotionController"]


_UR_DOF = 6


def _refused_by_the_halt(conn: object, what: str) -> bool:
    """Whether ``conn``'s halt latch refuses ``what``, nothing sent. Strict: a double that answers anything is not."""
    refuse = getattr(conn, "refuse_if_halted", None)
    return callable(refuse) and refuse(what) is True


def _ended_by_the_halt(conn: object) -> bool:
    """Whether the move ``conn`` was last asked for was refused or braked by the halt latch."""
    return getattr(conn, "last_move_end", None) in HALT_ENDS


class _URFKAdapter:
    """A small adapter that lets the singularity helpers consume the URConnection FK."""

    def __init__(self, conn: URConnection) -> None:
        self._conn = conn

    def fk(self, jp: JointPositions):
        tcp = self._conn.fk(jp.tolist())
        return urpose_to_pose(URPose.from_ur_list(tcp), frame=Frame.BASE)


class _URChainFKAdapter:
    """The singularity helpers' FK on the arm's own DH chain, with no round trip (``singularity_fk: dh``).

    ``chain`` is a :class:`~src.robot.safety._ur_kinematics.URDhChain`, the controller's calibrated rows or the model's
    nominal table, and ``tcp_m`` the controller's active TCP as a 4x4 in metres: the TCP pose ``getForwardKinematics``
    answers with that offset, where the controller computes on those rows.
    """

    def __init__(self, chain: object, tcp_m: "np.ndarray") -> None:
        self._chain = chain
        self._tcp_m = tcp_m

    def fk(self, jp: JointPositions) -> Pose:
        tcp = self._chain.flange_m(jp.tolist()) @ self._tcp_m  # type: ignore[attr-defined]
        tcp[:3, 3] *= 1000.0
        return Pose.from_matrix(tcp, frame=Frame.BASE)


class MotionController:
    """Safe high-level motion over one open UR connection.

    Parameters:
        connection: An open ``URConnection``.
        guard: The ``WorkspaceGuard`` that validates every target pose.
        max_velocity: A hard cap on the commanded velocity. ``None`` disables clamping.
        max_acceleration: A hard cap on the commanded acceleration.
        singularity_fk: ``robot.safety.ik_quality.singularity_fk``, which FK the singularity check before a move
            differentiates: ``"controller"``, 12 round trips as always, or ``"dh"``, the arm's own DH chain here.
        arm_model: ``robot.ur.model``, whose nominal table the ``"dh"`` check runs on where the controller's own rows
            cannot be read.
    """

    def __init__(
        self,
        connection: URConnection,
        guard: WorkspaceGuard,
        max_velocity: float | None = None,
        max_acceleration: float | None = None,
        singularity_thresholds: SingularityThresholds | None = None,
        *,
        singularity_fk: str = "controller",
        arm_model: str | None = None,
    ):
        self.conn = connection
        self.guard = guard
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.singularity_thresholds = singularity_thresholds or SingularityThresholds()
        if singularity_fk not in ("controller", "dh"):
            raise ValueError(f"singularity_fk is 'controller' or 'dh', got {singularity_fk!r}")
        self.singularity_fk = singularity_fk
        self.arm_model = arm_model
        #: What the ``"dh"`` check last said it runs on, so it says so once for every chain and TCP it meets; and whether
        #: the controller's FK is the nominal table's, once per connection (:meth:`_why_not_the_controllers_fk`).
        self._singularity_fk_said: object = None
        self._nominal_checked: "tuple[object, str] | None" = None
        self.logger = create_robot_logger("MotionController", UR_MOTION_LOG_FILE)
        # The typed cause of the most recent ``move_to`` that returned False, so
        # URRobotArm.move attaches the real MotionStatus rather than the generic
        # CONTROLLER_REJECTED default of from_bool. It is None on success and at the
        # start of every move_to. It holds for one controller instance, so one arm.
        self.last_reject_status: MotionStatus | None = None

    def _is_singularity_risky(self, joints: list[float]) -> bool:
        """Return ``True`` where the target joint state is near a singularity.

        The Jacobian is the central differences of a forward kinematics: the controller's, 12 round trips, or with
        :attr:`singularity_fk` ``"dh"`` the arm's own DH chain here (:meth:`_singularity_fk`), with the same
        differences and the same thresholds. On the controller's own rows the two are one FK: URSim CB3 answers them
        to 6e-15 m, nominal and calibrated alike.
        """

        report = analyze_joint_singularity(
            cast("RobotArm", self._singularity_fk()),
            JointPositions(joints),
            thresholds=self.singularity_thresholds,
        )
        if report.is_near_singularity:
            self.logger.warning(
                "Target rejected due to singularity risk: %s",
                "; ".join(report.reasons),
            )
            return True
        return False

    def _singularity_fk(self) -> "_URFKAdapter | _URChainFKAdapter":
        """The FK :meth:`_is_singularity_risky` differentiates: the controller's, or with :attr:`singularity_fk`
        ``"dh"`` the arm's own chain, its controller's calibrated rows where they can be read
        (``URConnection.controller_kinematics``, once per connection) and :attr:`arm_model`'s nominal table where they
        cannot and the controller's own FK confirms the table is its chain (:meth:`_why_not_the_controllers_fk`), times
        the active TCP (``URConnection.active_tcp_offset``). Anywhere else, the controller's, as always; which one is
        said once for every chain it meets.
        """
        if self.singularity_fk != "dh":
            return _URFKAdapter(self.conn)
        from src.robot.safety._ur_ik import ur_pose_matrix_m
        from src.robot.safety._ur_kinematics import URDhChain

        from .connection import ControllerKinematics

        read_rows = getattr(self.conn, "controller_kinematics", None)
        read_tcp = getattr(self.conn, "active_tcp_offset", None)
        try:
            kinematics = read_rows() if callable(read_rows) else None
            tcp = read_tcp() if callable(read_tcp) else None
            if isinstance(kinematics, ControllerKinematics):
                chain: "URDhChain | None" = URDhChain(theta_rad=kinematics.theta_rad, a_m=kinematics.a_m,
                                                      d_m=kinematics.d_m, alpha_rad=kinematics.alpha_rad)
                rows = f"the controller's own rows (read from port {kinematics.port})"
            else:
                chain = URDhChain.nominal(self.arm_model) if self.arm_model else None
                rows = f"the nominal {self.arm_model} table, the controller's own rows not read"
            tcp_m = ur_pose_matrix_m(tcp) if isinstance(tcp, list) else None
            if chain is not None and tcp_m is not None and not isinstance(kinematics, ControllerKinematics):
                why = self._why_not_the_controllers_fk(chain, tcp_m)
                if why:
                    chain, rows = None, f"the controller's own rows not read, and {why}"
        except (RuntimeError, OSError, ValueError) as exc:
            chain, tcp_m, rows = None, None, f"no chain: {exc}"
        if chain is None or tcp_m is None:
            said: object = ("controller", rows)
            if said != self._singularity_fk_said:
                self._singularity_fk_said = said
                self.logger.warning("The singularity check before a move asks the controller's FK, as with "
                                    "singularity_fk: controller: %s.", rows if chain is None else "no active TCP read")
            return _URFKAdapter(self.conn)
        said = (rows, tuple(float(v) for v in (tcp or ())))
        if said != self._singularity_fk_said:
            self._singularity_fk_said = said
            self.logger.info("The singularity check before a move runs on the arm's own DH chain, %s, times the active "
                             "TCP %s (robot.safety.ik_quality.singularity_fk: dh).", rows,
                             [round(float(v), 6) for v in (tcp or ())])
        return _URChainFKAdapter(chain, tcp_m)

    #: Two configurations of no particular pose, where a calibrated controller's FK and the nominal table's part.
    _NOMINAL_PROBES: tuple[tuple[float, ...], ...] = (
        (0.3, -1.2, 1.4, -0.8, 1.1, 0.4),
        (-1.3, -1.9, -1.0, -1.7, 1.6, -1.3),
    )
    #: How close, metres and radians, the controller's FK has to answer the nominal table's at both probes for the table
    #: to be the controller's chain: a calibration parts them by millimetres, a TCP of zero sent as 1 nm by 1e-9 m.
    _NOMINAL_AGREES = 1e-7

    def _why_not_the_controllers_fk(self, chain: object, tcp_m: "np.ndarray") -> str:
        """``""`` where the controller's own FK is the nominal ``chain`` times ``tcp_m``, else why not; asked once per
        connection (two round trips at :data:`_NOMINAL_PROBES`, kept by the connection's ``kinematics_epoch``).

        The nominal table stands in for rows that cannot be read only where it is the controller's chain, as on an
        uncalibrated arm or URSim: against a calibrated controller the check's verdict parted from the controller's near
        both thresholds on 38 of 156 configurations of lines toward the three singularities (URSim CB3 with a
        ``calibration.conf``), so there the controller is asked as always.
        """
        import numpy as np

        from src.robot.safety._ur_ik import ur_pose_matrix_m

        key = (id(self.conn), getattr(self.conn, "kinematics_epoch", None), chain, tcp_m.tobytes())
        if self._nominal_checked is not None and self._nominal_checked[0] == key:
            return self._nominal_checked[1]
        why = ""
        for probe in self._NOMINAL_PROBES:
            try:
                answer = ur_pose_matrix_m(self.conn.fk(list(probe)))
            except (RuntimeError, OSError, ValueError) as exc:
                why = f"the controller's FK could not be asked to confirm the nominal table ({exc})"
                break
            ours = chain.flange_m(probe) @ tcp_m  # type: ignore[attr-defined]
            apart_m = float(np.linalg.norm(answer[:3, 3] - ours[:3, 3]))
            apart_rad = float(np.linalg.norm(answer[:3, :3] - ours[:3, :3]))
            if not (apart_m <= self._NOMINAL_AGREES and apart_rad <= self._NOMINAL_AGREES):
                why = (f"its FK stands {apart_m * 1000.0:.3f} mm from the nominal table's, a calibration the nominal "
                       "table does not hold")
                break
        self._nominal_checked = (key, why)
        return why

    def _clamp(self, vel: float | None, acc: float | None) -> tuple[float | None, float | None]:
        """Clamp a caller-supplied velocity and acceleration to the configured hard limits."""
        if vel is not None and self.max_velocity is not None and vel > self.max_velocity:
            self.logger.warning(
                "Requested velocity %.3f exceeds limit %.3f, clamping.",
                vel, self.max_velocity,
            )
            vel = self.max_velocity
        if acc is not None and self.max_acceleration is not None and acc > self.max_acceleration:
            self.logger.warning(
                "Requested acceleration %.3f exceeds limit %.3f, clamping.",
                acc, self.max_acceleration,
            )
            acc = self.max_acceleration
        return vel, acc

    def get_current_pose(self) -> URPose:
        """Read the current TCP pose from the robot."""
        tcp = self.conn.get_tcp_pose()
        return URPose.from_ur_list(tcp, label="current")

    @staticmethod
    def _valid_ik_solution(joints: list[float]) -> bool:
        return bool(joints) and len(joints) == _UR_DOF

    def move_to(
        self,
        pose: URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
        workspace_pose: URPose | None = None,
    ) -> bool:
        """Move the robot to ``pose`` after workspace validation.

        ``pose`` is what the controller is told, in the frame its tool register uses.
        ``workspace_pose`` is what the workspace box judges when that differs: in
        ``willy`` tool-frame mode the controller is told the flange target, while
        ``workspace_limits`` bound the grasp centre. ``None`` boxes ``pose`` itself.
        """
        # Classify each rejection, so URRobotArm.move attaches the real typed cause
        # rather than the generic CONTROLLER_REJECTED of from_bool. It is cleared per
        # call and set at each branch below.
        self.last_reject_status = None
        if not self.conn.is_connected:
            self.logger.error("Cannot move: robot is not connected.")
            self.last_reject_status = MotionStatus.CONNECTION_ERROR
            return False
        if _refused_by_the_halt(self.conn, "moveL" if linear else "moveJ"):
            # Refused before the inverse kinematics and the singularity check read the controller: nothing at all.
            self.logger.error("Not moving to '%s': the arm is halted.", pose.label)
            self.last_reject_status = MotionStatus.CANCELLED
            return False

        # The workspace box alone, deliberately not `validate()`, which also runs the
        # pose-diversity check. That check, refusing a pose too similar to one already
        # accepted, belongs to calibration pose sampling, and that is where it lives:
        # `execution/pose_provider.py` builds its own local guard for it. Wired into the
        # general motion path it rejects the ordinary shape of a pick. Measured: the
        # standoff is accepted, the descend is accepted, and the retreat is rejected as
        # too similar to the standoff at 0.0 mm and 0.0 deg. The arm would grasp and
        # refuse to lift on every pick, and the symptom reads as a gripper, planner or
        # calibration fault rather than a sampling rule.
        boxed = pose if workspace_pose is None else workspace_pose
        if not self.guard.is_inside_workspace(boxed):
            self.logger.warning("Pose '%s' rejected by workspace guard.", pose.label)
            self.last_reject_status = MotionStatus.WORKSPACE_REJECTED
            return False

        vel, acc = self._clamp(vel, acc)

        tcp = pose.to_ur_list()
        try:
            joints = self.conn.ik(tcp)
        except (RuntimeError, OSError) as exc:
            self.logger.exception("IK failed for pose '%s': %s", pose.label, exc)
            self.last_reject_status = MotionStatus.IK_FAILED
            return False

        if not self._valid_ik_solution(joints):
            self.logger.error(
                "IK failed for pose '%s': no valid solution.", pose.label,
            )
            self.last_reject_status = MotionStatus.IK_FAILED
            return False

        try:
            if self._is_singularity_risky(joints):
                # There is real Jacobian evidence, so this surfaces as
                # IK_QUALITY_REJECTED, matching the preflight ik_quality mapping, rather
                # than as a generic controller rejection.
                self.last_reject_status = MotionStatus.IK_QUALITY_REJECTED
                return False
        except (RuntimeError, OSError, ValueError) as exc:
            self.logger.warning(
                "Singularity analysis unavailable for pose '%s': %s. Continuing move.",
                pose.label,
                exc,
            )
            # Keep the runtime behaviour where FK support is unavailable.

        self.logger.info(
            "Moving to '%s' [%.1f, %.1f, %.1f mm] (%s) ...",
            pose.label, pose.x, pose.y, pose.z,
            "linear" if linear else "joint",
        )

        try:
            if linear:
                ok = self.conn.moveL(tcp, vel=vel, acc=acc)
            else:
                ok = self.conn.moveJ(joints, vel=vel, acc=acc)
        except (RuntimeError, OSError) as exc:
            self.logger.exception("Move to '%s' failed: %s", pose.label, exc)
            self.last_reject_status = MotionStatus.CONTROLLER_REJECTED
            return False

        if ok and register:
            self.guard.accept(boxed)

        if not ok:
            # A move the halt refused or braked is CANCELLED, not a refusal of the controller's.
            self.last_reject_status = (MotionStatus.CANCELLED if _ended_by_the_halt(self.conn)
                                       else MotionStatus.CONTROLLER_REJECTED)
        return bool(ok)

    def move_home(self, home_joints=None) -> bool:
        """Move to a known safe home position in joint space."""
        if not self.conn.is_connected:
            self.logger.error("Cannot move home: robot is not connected.")
            return False
        if _refused_by_the_halt(self.conn, "moveJ"):
            self.logger.error("Not moving home: the arm is halted.")
            return False

        if home_joints is None:
            home_joints = list(HOME_JOINTS_DEFAULT)

        try:
            tcp = self.conn.fk(home_joints)
            home_pose = URPose.from_ur_list(tcp, label="home")
            if not self.guard.is_inside_workspace(home_pose):
                self.logger.warning(
                    "Home pose (%.1f, %.1f, %.1f) lies outside workspace; "
                    "proceeding anyway.",
                    home_pose.x, home_pose.y, home_pose.z,
                )
        except (RuntimeError, OSError, ValueError) as exc:
            self.logger.debug("FK for home check failed: %s", exc)

        self.logger.info("Moving to home position ...")
        try:
            return bool(self.conn.moveJ(home_joints))
        except (RuntimeError, OSError) as exc:
            self.logger.exception("Move home failed: %s", exc)
            return False

    async def amove_to(
        self,
        pose: URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
        workspace_pose: URPose | None = None,
    ) -> bool:
        """Async variant of :meth:`move_to` using ``asyncio.to_thread``."""
        return await asyncio.to_thread(
            self.move_to, pose,
            linear=linear, vel=vel, acc=acc, register=register, workspace_pose=workspace_pose,
        )

    async def amove_home(self, home_joints=None) -> bool:
        """Async variant of :meth:`move_home`."""
        return await asyncio.to_thread(self.move_home, home_joints)

    async def await_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Async variant of :meth:`URConnection.wait_until_steady`."""
        return await asyncio.to_thread(
            self.conn.wait_until_steady, timeout_s, poll_interval_s,
        )
