"""MotionController, the safe high-level UR motion execution.

It wraps ``URConnection`` with workspace-guard checks, so every Cartesian move is
validated before it reaches the controller.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from src.geometry import Frame
from src.robot.constants import HOME_JOINTS_DEFAULT, UR_MOTION_LOG_FILE, create_robot_logger
from src.robot.core import JointPositions, MotionStatus
from src.robot.drivers.ur import URConnection
from src.robot.drivers.ur.pose import URPose
from src.robot.drivers.ur.pose_adapter import urpose_to_pose
from src.robot.safety.singularity import (
    SingularityThresholds,
    analyze_joint_singularity,
)
from src.robot.safety.workspace import WorkspaceGuard

if TYPE_CHECKING:
    from src.robot.core import RobotArm

__all__ = ["MotionController"]


_UR_DOF = 6


class _URFKAdapter:
    """A small adapter that lets the singularity helpers consume the URConnection FK."""

    def __init__(self, conn: URConnection) -> None:
        self._conn = conn

    def fk(self, jp: JointPositions):
        tcp = self._conn.fk(jp.tolist())
        return urpose_to_pose(URPose.from_ur_list(tcp), frame=Frame.BASE)


class MotionController:
    """Safe high-level motion over one open UR connection.

    Parameters:
        connection: An open ``URConnection``.
        guard: The ``WorkspaceGuard`` that validates every target pose.
        max_velocity: A hard cap on the commanded velocity. ``None`` disables clamping.
        max_acceleration: A hard cap on the commanded acceleration.
    """

    def __init__(
        self,
        connection: URConnection,
        guard: WorkspaceGuard,
        max_velocity: float | None = None,
        max_acceleration: float | None = None,
        singularity_thresholds: SingularityThresholds | None = None,
    ):
        self.conn = connection
        self.guard = guard
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.singularity_thresholds = singularity_thresholds or SingularityThresholds()
        self.logger = create_robot_logger("MotionController", UR_MOTION_LOG_FILE)
        # The typed cause of the most recent ``move_to`` that returned False, so
        # URRobotArm.move attaches the real MotionStatus rather than the generic
        # CONTROLLER_REJECTED default of from_bool. It is None on success and at the
        # start of every move_to. It holds for one controller instance, so one arm.
        self.last_reject_status: MotionStatus | None = None

    def _is_singularity_risky(self, joints: list[float]) -> bool:
        """Return ``True`` where the target joint state is near a singularity."""

        report = analyze_joint_singularity(
            cast("RobotArm", _URFKAdapter(self.conn)),
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
    ) -> bool:
        """Move the robot to ``pose`` after workspace validation."""
        # Classify each rejection, so URRobotArm.move attaches the real typed cause
        # rather than the generic CONTROLLER_REJECTED of from_bool. It is cleared per
        # call and set at each branch below.
        self.last_reject_status = None
        if not self.conn.is_connected:
            self.logger.error("Cannot move: robot is not connected.")
            self.last_reject_status = MotionStatus.CONNECTION_ERROR
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
        if not self.guard.is_inside_workspace(pose):
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
            self.guard.accept(pose)

        if not ok:
            self.last_reject_status = MotionStatus.CONTROLLER_REJECTED
        return bool(ok)

    def move_home(self, home_joints=None) -> bool:
        """Move to a known safe home position in joint space."""
        if not self.conn.is_connected:
            self.logger.error("Cannot move home: robot is not connected.")
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
    ) -> bool:
        """Async variant of :meth:`move_to` using ``asyncio.to_thread``."""
        return await asyncio.to_thread(
            self.move_to, pose,
            linear=linear, vel=vel, acc=acc, register=register,
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
