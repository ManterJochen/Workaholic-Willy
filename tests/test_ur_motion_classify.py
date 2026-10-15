"""Gap L3.2 — MotionController.move_to classifies its rejection cause (last_reject_status).

So URRobotArm.move can attach the TRUE typed MotionStatus (a real near-singularity ->
IK_QUALITY_REJECTED) instead of MotionResult.from_bool's generic CONTROLLER_REJECTED default.
"""

from __future__ import annotations

import numpy as np

from src.config.schema.robot.robot_schema import WorkspaceLimitsConfig
from src.geometry import Frame, Pose
from src.robot.core import MotionStatus
from src.robot.drivers.ur.motion import MotionController
from src.robot.drivers.ur.pose import URPose
from src.robot.safety.workspace import WorkspaceGuard


class _FakeConn:
    def __init__(self, connected: bool = True) -> None:
        self._connected = connected

    @property
    def is_connected(self) -> bool:
        return self._connected

    def ik(self, tcp: list[float]) -> list[float]:
        return [0.0, -1.5, 1.5, 0.0, 1.5, 0.0]  # a valid 6-DoF solution

    def moveJ(self, joints: list[float], *, vel: float | None = None, acc: float | None = None) -> bool:
        return True

    def moveL(self, tcp: list[float], *, vel: float | None = None, acc: float | None = None) -> bool:
        return True


class _Guard:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok
        self.accepted: list[object] = []

    def is_inside_workspace(self, pose: object) -> bool:
        """The motion path checks the BOX only. The pose-diversity half of `validate()` belongs to
        calibration sampling (pose_provider builds its own guard for it) and, wired in here, rejected
        every pick's retreat as "too similar" to its own standoff."""
        return self._ok

    def validate(self, pose: object) -> bool:
        return self._ok

    def accept(self, pose: object) -> None:
        self.accepted.append(pose)


def _pose() -> URPose:
    return URPose(x=400.0, y=0.0, z=300.0, rx=0.0, ry=3.14159, rz=0.0, label="t")


def _controller(conn: _FakeConn | None = None, guard: _Guard | None = None) -> MotionController:
    return MotionController(conn or _FakeConn(), guard or _Guard(), max_velocity=1.0, max_acceleration=1.0)  # type: ignore[arg-type]


class TestMotionClassify:
    def test_not_connected_is_connection_error(self) -> None:
        mc = _controller(conn=_FakeConn(connected=False))
        assert mc.move_to(_pose()) is False
        assert mc.last_reject_status is MotionStatus.CONNECTION_ERROR

    def test_a_value_ur_rtde_refuses_before_sending_is_an_invalid_target(self) -> None:
        """P0g R3: ur_rtde raises ValueError for a value outside its range, a speed or an acceleration, before it sends
        anything. That is a refusal with nothing sent, typed INVALID_TARGET, not an exception out of the arm; the ik
        route, the judged line and move_linear all reach the controller through here."""

        class _Refusing(_FakeConn):
            def moveJ(self, joints: list[float], *, vel: float | None = None, acc: float | None = None) -> bool:
                raise ValueError("The value is not within [0;3.14]")

            def moveL(self, tcp: list[float], *, vel: float | None = None, acc: float | None = None) -> bool:
                raise ValueError("The value is not within [0;3.14]")

        for linear in (False, True):
            mc = _controller(conn=_Refusing())
            mc._is_singularity_risky = lambda joints: False  # type: ignore[method-assign]
            assert mc.move_to(_pose(), linear=linear) is False
            assert mc.last_reject_status is MotionStatus.INVALID_TARGET

    def test_workspace_reject_is_workspace_rejected(self) -> None:
        mc = _controller(guard=_Guard(ok=False))
        assert mc.move_to(_pose()) is False
        assert mc.last_reject_status is MotionStatus.WORKSPACE_REJECTED

    def test_singularity_is_ik_quality_rejected(self) -> None:
        # The ONLY branch with real Jacobian evidence must surface as IK_QUALITY_REJECTED, not the
        # generic CONTROLLER_REJECTED default — the heart of L3.2.
        mc = _controller()
        mc._is_singularity_risky = lambda joints: True  # type: ignore[method-assign]
        assert mc.move_to(_pose()) is False
        assert mc.last_reject_status is MotionStatus.IK_QUALITY_REJECTED

    def test_success_clears_status(self) -> None:
        mc = _controller()
        mc._is_singularity_risky = lambda joints: False  # type: ignore[method-assign]
        assert mc.move_to(_pose()) is True
        assert mc.last_reject_status is None


class TestTheRealBoxJudgesTheMove:
    """The doubles above take any object, so they would pass a URPose straight through. The real guard speaks
    Pose only since P0e and raises on anything else: this is the class that proves the controller converts
    before it asks, and asks about the pose it names."""

    @staticmethod
    def _real(z_max: float = 500.0) -> tuple[MotionController, WorkspaceGuard]:
        guard = WorkspaceGuard(WorkspaceLimitsConfig(z_max=z_max))
        mc = MotionController(_FakeConn(), guard, max_velocity=1.0, max_acceleration=1.0)  # type: ignore[arg-type]
        mc._is_singularity_risky = lambda joints: False  # type: ignore[method-assign]
        return mc, guard

    def test_inside_the_box_moves_and_the_box_keeps_a_base_frame_pose(self) -> None:
        mc, guard = self._real()
        assert mc.move_to(_pose()) is True
        assert mc.last_reject_status is None
        (kept,) = guard.accepted_poses
        assert isinstance(kept, Pose) and kept.frame is Frame.BASE and kept.label == "t"
        np.testing.assert_allclose(kept.position_mm, [400.0, 0.0, 300.0], atol=1e-12)

    def test_outside_the_box_is_refused_and_nothing_is_kept(self) -> None:
        mc, guard = self._real(z_max=250.0)
        assert mc.move_to(_pose()) is False
        assert mc.last_reject_status is MotionStatus.WORKSPACE_REJECTED
        assert guard.accepted_poses == []

    def test_the_box_judges_the_workspace_pose_when_one_is_given(self) -> None:
        """In ``willy`` mode the controller is told the flange and the box bounds the grasp centre."""
        mc, guard = self._real(z_max=250.0)
        centre = URPose(x=400.0, y=0.0, z=200.0, rx=0.0, ry=3.14159, rz=0.0, label="centre")
        assert mc.move_to(_pose(), workspace_pose=centre) is True
        (kept,) = guard.accepted_poses
        np.testing.assert_allclose(kept.position_mm, [400.0, 0.0, 200.0], atol=1e-12)
        mc2, _ = self._real(z_max=250.0)
        assert mc2.move_to(_pose(), workspace_pose=_pose()) is False
