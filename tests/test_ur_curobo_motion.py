"""Hardware-free unit tests for the real-UR cuRobo motion glue (``CuroboUrPlanner``).

The cuRobo round-trip on a real robot + real GPU env is bucket-③ (not on CI). Here we
inject a fake cuRobo client + a fake UR connection and lock the driver LOGIC: goal
conversion (mm→m, XYZW→WXYZ), joint-order remap (planner ↔ UR), waypoint execution,
and every fail-closed branch (no env / no plan / not connected / moveJ reject / raise).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry import Frame, Pose
from src.robot.core import MotionStatus
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES, CuroboUrPlanner
from src.robot.safety.planning import CuroboUnavailableError, JointCheckVerdict


class _FakeClient:
    def __init__(self, joint_names, traj, *, unavailable=False):
        self.joint_names = list(joint_names)
        self.dt = 0.0
        self._traj = traj
        self._unavailable = unavailable
        self.started = False
        self.closed = False
        self.plan_args = None
        self.world = None
        self.calls: list[str] = []
        self.checked: list[list[float]] | None = None
        self._verdict = JointCheckVerdict(
            valid=True, first_invalid=None, checked=0, reason="the fake accepts"
        )

    def start(self):
        self.calls.append("start")
        if self._unavailable:
            raise CuroboUnavailableError("no cuRobo env")
        self.started = True

    def plan(self, start, pos_m, quat_wxyz):
        self.calls.append("plan")
        self.plan_args = (list(start), list(pos_m), list(quat_wxyz))
        return self._traj

    def check_joints(self, configs):
        self.calls.append("check_joints")
        self.checked = [list(c) for c in configs]
        return self._verdict

    def set_world(self, cuboids):
        self.calls.append("set_world")
        self.world = list(cuboids)
        return len(cuboids)

    def close(self):
        self.closed = True


class _FakeConn:
    def __init__(self, joints, *, connected=True, move_ok=True, raise_move=False):
        self._joints = list(joints)
        self._connected = connected
        self._move_ok = move_ok
        self._raise = raise_move
        self.moves: list[list[float]] = []

    @property
    def is_connected(self):
        return self._connected

    def get_joint_positions(self):
        return list(self._joints)

    def moveJ(self, joints, vel=None, acc=None):
        if self._raise:
            raise RuntimeError("moveJ boom")
        self.moves.append([float(v) for v in joints])
        return self._move_ok


def _pose(x=100.0, y=200.0, z=300.0):
    return Pose(
        position_mm=np.array([x, y, z], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        frame=Frame.BASE,
        label="goal",
    )


def _planner(conn, client):
    return CuroboUrPlanner(conn, client_factory=lambda: client)


def test_move_converts_goal_and_executes_waypoints() -> None:
    traj = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
    client = _FakeClient(UR_ARM_JOINT_NAMES, traj)
    conn = _FakeConn([1, 2, 3, 4, 5, 6])
    result = _planner(conn, client).move(_pose())

    assert result.status is MotionStatus.EXECUTED
    # goal: mm→m and XYZW[0,0,0,1] → WXYZ[1,0,0,0].
    start, pos_m, quat_wxyz = client.plan_args
    assert pos_m == pytest.approx([0.1, 0.2, 0.3])
    assert quat_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert start == pytest.approx([1, 2, 3, 4, 5, 6])  # identity remap (names == UR order)
    # both waypoints executed via moveJ, in order.
    assert conn.moves == [pytest.approx(traj[0]), pytest.approx(traj[1])]


def test_joint_order_remap_when_client_names_permuted() -> None:
    # Planner reports joints in REVERSED UR order → start + traj must be remapped.
    permuted = list(reversed(UR_ARM_JOINT_NAMES))
    traj = [[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]  # in permuted (reversed) order
    client = _FakeClient(permuted, traj)
    conn = _FakeConn([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # UR order
    result = _planner(conn, client).move(_pose())

    assert result.status is MotionStatus.EXECUTED
    # start sent to the planner is the UR-order current joints reordered into planner (reversed) order.
    start, _, _ = client.plan_args
    assert start == pytest.approx([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    # executed waypoint is the planner-order traj reordered back to UR order.
    assert conn.moves == [pytest.approx([15.0, 14.0, 13.0, 12.0, 11.0, 10.0])]


def test_fail_closed_when_env_unavailable() -> None:
    client = _FakeClient(UR_ARM_JOINT_NAMES, [], unavailable=True)
    conn = _FakeConn([0] * 6)
    result = _planner(conn, client).move(_pose())
    assert result.status is MotionStatus.CONTROLLER_REJECTED
    assert conn.moves == []  # no blind motion


def test_fail_closed_when_no_plan() -> None:
    client = _FakeClient(UR_ARM_JOINT_NAMES, None)  # cuRobo found nothing
    conn = _FakeConn([0] * 6)
    result = _planner(conn, client).move(_pose())
    assert result.status is MotionStatus.TIMEOUT
    assert conn.moves == []


def test_fail_when_not_connected() -> None:
    client = _FakeClient(UR_ARM_JOINT_NAMES, [[0] * 6])
    conn = _FakeConn([0] * 6, connected=False)
    result = _planner(conn, client).move(_pose())
    assert result.status is MotionStatus.CONNECTION_ERROR


def test_reject_when_movej_returns_false() -> None:
    client = _FakeClient(UR_ARM_JOINT_NAMES, [[0] * 6])
    conn = _FakeConn([0] * 6, move_ok=False)
    result = _planner(conn, client).move(_pose())
    assert result.status is MotionStatus.CONTROLLER_REJECTED


def test_connection_error_when_movej_raises() -> None:
    client = _FakeClient(UR_ARM_JOINT_NAMES, [[0] * 6])
    conn = _FakeConn([0] * 6, raise_move=True)
    result = _planner(conn, client).move(_pose())
    assert result.status is MotionStatus.CONNECTION_ERROR


def test_invalid_frame_rejected() -> None:
    client = _FakeClient(UR_ARM_JOINT_NAMES, [[0] * 6])
    conn = _FakeConn([0] * 6)
    bad = Pose.identity(Frame.CAMERA, label="cam")
    result = _planner(conn, client).move(bad)
    assert result.status is MotionStatus.INVALID_TARGET


def test_set_world_and_close() -> None:
    client = _FakeClient(UR_ARM_JOINT_NAMES, [[0] * 6])
    conn = _FakeConn([0] * 6)
    planner = _planner(conn, client)
    n = planner.set_world([{"name": "wall", "dims_m": [1, 1, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}])
    assert n == 1
    assert client.world is not None
    planner.close()
    assert client.closed is True


# --------------------------------------------------------------------------------------------
# The joint path, checked by the same planner that plans the Cartesian ones.
# --------------------------------------------------------------------------------------------


def test_a_joint_path_is_checked_against_the_registered_world() -> None:
    """Start, register the declared world, then ask. Nothing is sent to the controller."""
    client = _FakeClient(UR_ARM_JOINT_NAMES, [])
    conn = _FakeConn([1, 2, 3, 4, 5, 6])
    bench = [{"name": "bench", "dims": [1.0, 1.0, 0.1], "pose": [0, 0, 0, 1, 0, 0, 0]}]
    planner = CuroboUrPlanner(
        conn, client_factory=lambda: client, world_cuboids=bench, require_registration=False
    )
    verdict = planner.check_joint_path([[0.0] * 6, [0.1] * 6])

    assert verdict.valid
    assert client.calls == ["start", "set_world", "check_joints"], client.calls
    assert client.world == bench, "the check ran against a world the planner never got"
    assert conn.moves == [], "checking a path commanded a motion"
    assert client.checked == [[0.0] * 6, [0.1] * 6]


def test_the_samples_are_remapped_into_planner_order() -> None:
    """The same remap `plan` does, in the same direction, on every sample."""
    permuted = list(reversed(UR_ARM_JOINT_NAMES))
    client = _FakeClient(permuted, [])
    conn = _FakeConn([0.0] * 6)
    _planner(conn, client).check_joint_path([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])

    assert client.checked == [[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]


def test_an_unavailable_env_raises_rather_than_answering() -> None:
    """Fail closed, exactly as `plan` does: no verdict is better than a made up one."""
    client = _FakeClient(UR_ARM_JOINT_NAMES, [], unavailable=True)
    with pytest.raises(CuroboUnavailableError):
        _planner(_FakeConn([0.0] * 6), client).check_joint_path([[0.0] * 6])


def test_a_refusal_carries_the_index_the_sidecar_gave() -> None:
    client = _FakeClient(UR_ARM_JOINT_NAMES, [])
    client._verdict = JointCheckVerdict(
        valid=False, first_invalid=7, checked=20, reason="the cuRobo check refuses sample 7 of 20"
    )
    verdict = _planner(_FakeConn([0.0] * 6), client).check_joint_path([[0.0] * 6] * 20)

    assert (verdict.valid, verdict.first_invalid, verdict.checked) == (False, 7, 20)
    assert "sample 7 of 20" in verdict.reason


def test_an_empty_path_is_not_sent() -> None:
    """Nothing to judge, and the sidecar refuses an empty request anyway."""
    client = _FakeClient(UR_ARM_JOINT_NAMES, [])
    verdict = _planner(_FakeConn([0.0] * 6), client).check_joint_path([])

    assert verdict.valid and verdict.checked == 0
    assert "check_joints" not in client.calls
