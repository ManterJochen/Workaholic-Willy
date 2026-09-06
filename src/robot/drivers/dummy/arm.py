"""The dummy arm driver, a pure-Python implementation of :class:`src.robot.core.RobotArm`.

It exists to exercise a pipeline or a planner without real hardware, to develop one
offline, and to drive the vendor-neutral surface itself.

The driver keeps an in-memory ``(joints, tcp_pose)`` state and has no kinematic model:
``move_joint`` records the new joint vector, ``move_linear`` records the new pose,
``fk`` returns the last recorded pose and ``ik`` returns the last recorded joints. That
is deliberate, and a pipeline that needs real kinematics runs against the UR driver or
the Isaac-backed ``sim`` driver instead.
"""

from __future__ import annotations

import numpy as np

from src.geometry import Frame, FrameMismatchError, Pose
from src.geometry.quaternion import IDENTITY_QUAT_XYZW

from ...core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotMotionRejected,
)

__all__ = ["DUMMY_CAPABILITIES", "DummyRobotArm"]


DUMMY_CAPABILITIES = RobotCapabilities(
    vendor="dummy",
    model="dummy-6dof",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=False,
    has_native_ik=False,
    has_force_control=False,
    is_simulated=True,
)


class DummyRobotArm(RobotArm):
    """A lightweight in-memory arm satisfying :class:`RobotArm`.

    Parameters
    ----------
    dof : int
        Degrees of freedom for the synthetic arm. The default of 6 is UR-shaped.
    initial_pose : Pose, optional
        The TCP pose :meth:`get_tcp_pose` returns until something else is commanded. It
        must be in :attr:`Frame.BASE`, and defaults to ``(400, 0, 300)`` mm with the
        identity orientation.
    """

    def __init__(
        self,
        *,
        dof: int = 6,
        initial_pose: Pose | None = None,
    ) -> None:
        self._capabilities = (
            DUMMY_CAPABILITIES
            if dof == DUMMY_CAPABILITIES.dof
            else RobotCapabilities(
                vendor="dummy", model=f"dummy-{dof}dof", dof=dof,
                is_simulated=True,
                supports_joint_move=True, supports_linear_move=True,
                supports_async_move=False,
                has_native_fk=False, has_native_ik=False,
                has_force_control=False,
            )
        )
        self._connected = False
        self._joints = JointPositions(np.zeros(dof, dtype=np.float64))
        if initial_pose is None:
            initial_pose = Pose(
                position_mm=np.array([400.0, 0.0, 300.0]),
                quaternion_xyzw=IDENTITY_QUAT_XYZW,
                frame=Frame.BASE,
                label="dummy-home",
            )
        if initial_pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm initial_pose must be in Frame.BASE; got {initial_pose.frame!r}."
            )
        self._tcp = initial_pose
        self._home_pose = initial_pose

    # ---- introspection ----

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._capabilities

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ---- lifecycle ----

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    # ---- state ----

    def get_tcp_pose(self) -> Pose:
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        return self._tcp

    def get_joint_positions(self) -> JointPositions:
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        return self._joints

    # ---- safety ----

    @property
    def safety_preflight(self) -> None:
        """Always ``None``: the dummy gates nothing, and says so rather than staying silent.

        This is an answer and not an absence, and the difference is the point.
        Implementing :class:`~src.robot.safety.attestation.SafetyGated` and returning
        ``None`` reports ``UNGATED``, a stated decision. Not implementing it at all
        reports ``UNSTATED``, meaning nobody said. Both are unsafe to drive and an
        operator has to be able to tell them apart.

        This arm is why the attestation exists. A script that prints a cleared-safety
        line while driving this class, whose `move_to_joints` below simply drives, is
        reporting a run in which no guard ran, which `scripts/examples/03_pick.py:7-14`
        records. A run on the dummy is a wiring rehearsal and never a safety
        demonstration, and this is where it says so about itself.
        """
        return None

    # ---- motion ----

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """The typed joint move. The dummy carries no preflight, so it simply drives."""
        self.move_joint(joints, velocity=velocity, acceleration=acceleration)
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
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        if joints.dof != self._capabilities.dof:
            raise RobotMotionRejected(
                f"DummyRobotArm expected {self._capabilities.dof} DoF, got {joints.dof}."
            )
        self._joints = joints

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm.move_linear requires Frame.BASE; got {pose.frame!r}."
            )
        self._tcp = pose

    def stop(self) -> None:
        # There is no asynchronous motion in flight, so there is nothing to interrupt.
        pass

    # ---- kinematics (stubbed) ----

    def fk(self, joints: JointPositions) -> Pose:
        # The joints are recorded and not used: FK returns the last commanded pose.
        if joints.dof != self._capabilities.dof:
            raise RobotMotionRejected(
                f"DummyRobotArm.fk expected {self._capabilities.dof} DoF, got {joints.dof}."
            )
        return self._tcp

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm.ik requires Frame.BASE; got {pose.frame!r}."
            )
        return self._joints

    # ---- high-level pipeline surface ------------------------------------

    def is_inside_workspace(self, pose: Pose) -> bool:
        """The dummy has no workspace box, so it accepts every BASE-frame pose."""
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm.is_inside_workspace requires Frame.BASE; got {pose.frame!r}."
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
        """Record ``pose`` as the new TCP pose. It always succeeds.

        ``register`` is accepted for Protocol compatibility and ignored, because this
        driver keeps no diversity history.
        """
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm.move_to requires Frame.BASE; got {pose.frame!r}."
            )
        self._tcp = pose
        return True

    def move_home(self) -> bool:
        """Reset to the configured home TCP pose. It always succeeds."""
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        self._tcp = self._home_pose
        self._joints = JointPositions(np.zeros(self._capabilities.dof, dtype=np.float64))
        return True

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """A pure-Python sim has no motion in flight, so settling is instant."""
        return True

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """The typed counterpart of :meth:`move_to`.

        This driver has no workspace box and no real controller, so a successful move
        always returns :attr:`MotionStatus.EXECUTED`. A connection or frame fault is
        reported as a typed failure rather than raised.
        """
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"DummyRobotArm.move requires Frame.BASE; got {pose.frame!r}",
            )
        if not self._connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="DummyRobotArm is not connected",
            )
        self._tcp = pose
        return MotionResult.executed(
            MotionCommand.MOVE_TO, target_pose=pose,
        )
