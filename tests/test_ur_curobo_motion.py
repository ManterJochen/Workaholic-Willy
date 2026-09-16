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


def test_the_carried_part_is_sent_where_the_arm_places_it_in_tool0_metres() -> None:
    """The planner takes a centre in tool0, not a distance along +Z: the hand approaches along +Y (Step 4g.0)."""

    class _AttachingClient(_FakeClient):
        def attach_payload(self, joints, dims_m, pose):
            self.calls.append("attach")
            self.attached = (list(joints), list(dims_m), list(pose))
            return True

    client = _AttachingClient(UR_ARM_JOINT_NAMES, [[0.0] * 6])
    planner = CuroboUrPlanner(_FakeConn([0.0] * 6), client_factory=lambda: client, require_registration=False)
    assert planner.attach_payload([0.0] * 6, (70.0, 120.0, 70.0), (0.0, 212.0, 0.0)) is True
    _, dims_m, pose = client.attached
    assert dims_m == pytest.approx([0.07, 0.12, 0.07])
    assert pose == pytest.approx([0.0, 0.212, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_a_refused_world_refresh_is_logged_once_before_the_refusal() -> None:
    """The UR planner raised on a refused refresh and wrote nothing; only a successful one logged.

    The caller turns the raise into CONTROLLER_REJECTED, so the one place the reason is readable by a
    person is this log line, and it has to exist exactly once per refusal.
    """
    import unittest

    from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld
    from src.robot.safety.planning.perceived import WorldBuildLimits

    # A sighted camera and an arm that cannot place its own links: an ordinary refusal, not a camera
    # fault. Written first with a blind camera, which from Step 4e raises CameraWorldUnavailable instead
    # (tests/test_live_planner_world.py FreshFrameAttemptsTests), so the refusal branch is held here by a
    # reason nobody asks about twice.
    class _SightedCamera:
        def grab_surface_depth(self):
            return DepthSnapshot(
                depth_mm=np.full((40, 40), 1000.0, dtype=np.float64),
                intrinsics=np.array([[50.0, 0.0, 20.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]]),
                timestamp=__import__("time").time(),
            )

    world = LivePlannerWorld(
        cameras=(CameraView(name="overhead", depth_source=_SightedCamera(), camera_to_base=np.eye(4)),),
        declared=(),
        limits=WorldBuildLimits(
            x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0), support_plane_top_mm=0.0
        ),
    )
    client = _FakeClient(UR_ARM_JOINT_NAMES, [[0.0] * 6])
    planner = CuroboUrPlanner(
        _FakeConn([0.0] * 6), client_factory=lambda: client, live_world=world,
        self_envelope=lambda: None,
    )

    checker = unittest.TestCase()
    with checker.assertLogs("CuroboUrPlanner", level="ERROR") as logs:
        with pytest.raises(CuroboUnavailableError, match="its own links"):
            planner._refresh_world()  # noqa: SLF001

    errors = [record for record in logs.records if record.levelname == "ERROR"]
    assert len(errors) == 1, logs.output
    assert "its own links" in errors[0].getMessage()


def test_a_camera_that_stays_silent_raises_logged_once_and_clears_the_guard() -> None:
    """Owner, Step 4: after its fresh-frame attempts a silent camera raises out of the planner.

    Logged once here, because this is where the camera and the attempts are known. And the path guard
    is told the cell is empty of perceived boxes, as it is on any other refused refresh: a guard left
    holding boxes from a camera nobody can vouch for is checking a cell that no longer exists.
    """
    import unittest

    from src.robot.core.errors import CameraWorldUnavailable
    from src.robot.safety.planning.live_world import CameraView, LivePlannerWorld
    from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope, WorldBuildLimits

    class _SilentCamera:
        def __init__(self) -> None:
            self.grabs = 0

        def grab_surface_depth(self):
            self.grabs += 1
            return None

    camera = _SilentCamera()
    world = LivePlannerWorld(
        cameras=(CameraView(name="overhead", depth_source=camera, camera_to_base=np.eye(4)),),
        declared=(),
        limits=WorldBuildLimits(
            x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0), support_plane_top_mm=0.0
        ),
        fresh_frame_attempts=2,
    )
    told: list = []
    planner = CuroboUrPlanner(
        _FakeConn([0.0] * 6), client_factory=lambda: _FakeClient(UR_ARM_JOINT_NAMES, [[0.0] * 6]),
        live_world=world,
        self_envelope=lambda: SelfEnvelope(
            frames_mm=(np.eye(4),),
            capsules=(LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 300.0), radius_mm=90.0),),
        ),
        on_perceived_obstacles=told.append,
    )

    checker = unittest.TestCase()
    with checker.assertLogs("CuroboUrPlanner", level="ERROR") as logs:
        with pytest.raises(CameraWorldUnavailable):
            planner._refresh_world()  # noqa: SLF001

    errors = [record for record in logs.records if record.levelname == "ERROR"]
    assert len(errors) == 1, logs.output
    assert "overhead" in errors[0].getMessage()
    assert camera.grabs == 3
    assert told == [()]


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


def test_a_refused_plan_says_which_links_touched_and_still_fails_safe() -> None:
    """B1 S15: the sidecar's typed reason reaches the UR planner's own log, and the verdict does not change.

    A pick that stops here is a TIMEOUT with one fail safe sentence. Which two links the planner could not get past
    was inside the sidecar, so an operator read "no plan" and had nowhere to go next.
    """
    from unittest import mock

    from src.robot.safety.planning import StateRefusal, StateRefusalKind, StateWhere

    client = _FakeClient(UR_ARM_JOINT_NAMES, None)
    client.last_refusal = StateRefusal(
        where=StateWhere.START, kind=StateRefusalKind.SELF_COLLISION, joints=(0.0,) * 6,
        link_a="hand", link_b="wrist_1_link", depth_mm=12.2,
    )
    conn = _FakeConn([0] * 6)
    planner = _planner(conn, client)

    with mock.patch.object(planner.logger, "warning") as warned:
        result = planner.move(_pose())

    assert result.status is MotionStatus.TIMEOUT
    assert conn.moves == []
    assert planner.last_refusal is client.last_refusal
    said = " ".join(str(call) for call in warned.call_args_list)
    assert "wrist_1_link" in said and "12.2" in said, said


def test_a_client_that_names_no_refusal_is_the_planner_it_always_was() -> None:
    """The control: every injected client in this suite predates the typed reason, and none of them may break."""
    client = _FakeClient(UR_ARM_JOINT_NAMES, None)
    assert not hasattr(client, "last_refusal")
    planner = _planner(_FakeConn([0] * 6), client)
    assert planner.move(_pose()).status is MotionStatus.TIMEOUT
    assert planner.last_refusal is None


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
