"""Gap L3.2 — MotionController.move_to classifies its rejection cause (last_reject_status).

So URRobotArm.move can attach the TRUE typed MotionStatus (a real near-singularity ->
IK_QUALITY_REJECTED) instead of MotionResult.from_bool's generic CONTROLLER_REJECTED default.
"""

from __future__ import annotations

from src.robot.core import MotionStatus
from src.robot.drivers.ur.motion import MotionController
from src.robot.drivers.ur.pose import URPose


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
